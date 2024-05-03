import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import sparse
from loguru import logger
from sklearn.model_selection import train_test_split
from utils.plotter import Plotter
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .artifacts import ExperimentArtifacts


@dataclass(eq=False, order=False)
class Experiment(ABC):
    model: Any
    dataset: Any
    artifacts_handler: ExperimentArtifacts
    hyperparameters: Dict[str, Any]

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self):
        pass


@dataclass(eq=False, order=False)
class AttentionExperiment(Experiment):
    run_tag: str

    def __post_init__(self):
        self.test_size = self.hyperparameters.pop('test_size', 0.2)
        self.chunk_size = self.hyperparameters.pop('chunksize', 20000)

    def __split_data(self):

        tr_idx, self.test_idx, y_train, _ = train_test_split(
            np.arange(self.dataset.df_dynamic.shape[0]),
            self.dataset.y,
            test_size=self.test_size,
            stratify=self.dataset.y,
        )
        self.tr_idx, self.val_idx, y_train, _ = train_test_split(
            np.arange(tr_idx.shape[0]),
            y_train,
            test_size=self.test_size,
            stratify=y_train,
        )
        return y_train

    def setup(self):
        logger.info(f"Setting up Experiment {self.run_tag} \
            for model {self.model.model_id}.")
        y_train = self.__split_data()
        neg = y_train.shape[0] - y_train.sum()
        pos = y_train.sum()
        total = y_train.shape[0]
        weight_for_0 = (1 / neg)*(total)/2.0
        weight_for_1 = (1 / pos)*(total)/2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print('Weight for class 0 - Non_conversion path: {:.2f}'.format(
            weight_for_0
        ))
        print('Weight for class 1 - Conversion path: {:.2f}'.format(
            weight_for_1
        ))
        self.hyperparameters.update(
            {
                'class_weight': class_weight,
                'seq_maxlen': self.dataset.seq_maxlen,
                "cus_dep_shape": self.dataset.X_control.shape[-1],
            }
        )

    def fit(self):
        return self.model.fit(
            df_dynamic=self.dataset.df_dynamic,
            X_control=self.dataset.X_control,
            y=self.dataset.y,
            tr_idx=self.tr_idx,
            val_idx=self.val_idx,
            hyperparameters=self.hyperparameters,
        )

    def evaluate_model(self):
        X_path_test = self.model.X_path[self.test_idx]
        X_TD_test = self.model.X_TD[self.test_idx]
        X_control_test = self.dataset.X_control[self.test_idx]
        y_test = self.dataset.y[self.test_idx]
        results = self.model.model.evaluate(
            [X_path_test, X_TD_test, X_control_test],
            y_test,
            batch_size=self.hyperparameters['BATCH_SIZE'],
            verbose=0,
        )
        y_pred, att_weights = self.model.predict(self.test_idx)
        return results, att_weights, y_test, y_pred
# IMPORTANT FUNCTIONS

    def __pad_str_corpus(self, lst):
        return (lst + (self.dataset.seq_maxlen - len(lst)) * [''])

    def __duplicates(self, lst, item):
        return [i for i, x in enumerate(lst) if x == item]

    def generate_removal_effect(self):
        # generating all_df_important_medium
        X_path_test = self.model.X_path
        corpus_test = self.model.tokenizer.sequences_to_texts(X_path_test)
        all_medium = list(
            set(
                [
                    pad for lst in corpus_test
                    for pad in self.__pad_str_corpus(lst.split(' '))
                ]
            )
        )
        all_medium.remove(all_medium[0])
        logger.debug(f'Len_all_medium: {len(all_medium)}')
        num_words = self.model.tokenizer.num_words
        tk_medium = list(self.model.tokenizer.word_index.items())[:num_words]
        tk_medium = list(dict(tk_medium).keys())
        logger.debug(f'Len_tk_medium: {len(tk_medium)}')
        all_important_medium = [elt for elt in all_medium if elt in tk_medium]
        logger.debug(f'Len_all_imp_tk_medium: {len(all_important_medium)}')
        min_occurrences = self.hyperparameters['min_utm_occurrences']
        df_utm_valcounts = self.dataset.df_dynamic['utm_hash'] \
            .explode().value_counts()
        num_imp_medium = df_utm_valcounts.shape[0]
        logger.debug(f'Number of Important medium = {num_imp_medium}')
        important_df_utm = df_utm_valcounts[
            df_utm_valcounts > min_occurrences
        ].index.tolist()
        num_imp_medium = len(important_df_utm)
        logger.debug(f'Important mediuns with more than \
            {min_occurrences} occurences: {num_imp_medium}')
        all_df_important_medium = [
            elt for elt in all_important_medium if elt in important_df_utm
            ]
        # self.model.tokenizer.sequences_to_texts(X_path_test)
        df_medium_len = len(all_df_important_medium)
        logger.debug(f'Number of mediuns for removal effect calculation =\
             {df_medium_len}')
        # mounting dicts of tri2hash and hash2tri
        df_s = self.dataset.df_dynamic.reset_index(drop=True)
        df_con = self.dataset.df_control.reset_index(drop=True)
        df_exp = df_s[['utm_hash', 'trinomials']]
        exp_trinomials = df_exp['trinomials'].explode()
        df_exp = df_exp.explode('utm_hash')
        df_exp['exp_trinomials'] = exp_trinomials
        del exp_trinomials
        tri2hash = df_exp.groupby(['exp_trinomials'])['utm_hash']\
            .apply(list).apply(set).apply(list)  # .to_dict()
        hash2tri = tri2hash.to_frame().reset_index().explode('utm_hash')\
            .set_index('utm_hash')
        hash2tri = hash2tri.reset_index().drop_duplicates(subset='utm_hash')\
            .set_index('utm_hash')
        del df_exp
        # calculating removal effect
        rm_effect = (
            pd.DataFrame(
                index=all_df_important_medium,
                columns={'rm_eff', 'n_occ', 'trinomials'}
                )
            .reset_index()
            .rename(columns={'index': 'utm_hash'})
        )
        for mtest in all_df_important_medium:
            #         rm_effect.loc[(mtest,is_control)]['trinomials']=hash2tri.loc[mtest]['exp_trinomials']
            idx_m_isc_os = rm_effect[(rm_effect['utm_hash'] == mtest)].index
            rm_effect.loc[idx_m_isc_os, 'trinomials'] = \
                hash2tri.loc[mtest]['exp_trinomials']
            #######
            df_s['rm'] = (
                df_s[
                    (df_s['utm_hash'].apply(lambda x: x.count(mtest)) > 0)
                ]
                ['utm_hash']
                .apply(lambda x: list(self.__duplicates(x, mtest))[::-1])
            )
            dft = df_s.query('rm.notnull().values')
            y_true0 = (dft['is_conversion'] > 0).values.astype(int)
        #             dftc=df_con.loc[is_control_index].loc[df_s.query('rm.notnull().values').index]
            dftc = df_con.loc[df_s.query('rm.notnull().values').index]
            df_s = df_s.drop(columns='rm', axis=1)
            corpus_test = dft['utm_hash'].to_list()
            X_test = self.model.tokenizer.texts_to_sequences(corpus_test)
            X_test = pad_sequences(
                X_test,
                maxlen=self.dataset.seq_maxlen,
                padding='post'
                )  # customer path
            TD_test = pad_sequences(
                dft['sess'].values,
                maxlen=self.dataset.seq_maxlen,
                dtype='float32',
                padding='post'
                )  # time decay from sessions
            # to C_test to have same dimensions of C_train
            C_test = pd.DataFrame(columns=self.dataset.Ccols)
            C_test = C_test.append(pd.get_dummies(
                dftc.drop('gid', axis=1),
                columns=['country_region', 'gender']
                ))
            C_test = C_test.fillna(0).values
            if C_test.shape[0] > 0:
                y_pred_i = self.model.model.predict(
                    [X_test, TD_test, C_test],
                    batch_size=self.hyperparameters['BATCH_SIZE']
                    )
                to_rm = dft['rm'].values
                try:
                    token = self.model.tokenizer.word_index[mtest]
                except Exception:
                    token = 0
                if token > self.model.tokenizer.num_words:
                    token = 0
                token_list = np.arange(0, token).tolist() + \
                    np.arange(token+1, self.model.tokenizer.num_words).tolist()
                for ii in range(X_test.shape[0]):
                    X_test[ii][to_rm[ii]] = \
                        np.random.choice(token_list, len(to_rm[ii]))
                if C_test.shape[0] > 0:
                    y_pred_rm = self.model.model.predict(
                        [X_test, TD_test, C_test],
                        batch_size=self.hyperparameters['BATCH_SIZE'])
                    rm_effect.loc[idx_m_isc_os, 'n_occ'] = y_true0.shape[0]
                    if any(y_true0 == 1):
                        rm_val = (np.median(y_pred_rm[y_true0 == 1]) -
                                  np.median(y_pred_i[y_true0 == 1]))\
                            / np.median(y_pred_i[y_true0 == 1])
                        if ~np.isnan(rm_val):
                            rm_effect.loc[idx_m_isc_os, 'rm_eff'] = rm_val
        rm_effect['scaled_rm_eff'] = rm_effect['n_occ']*rm_effect['rm_eff']
        #  GROUPING BY TRINOMIALS
        df_tri_re = rm_effect\
            .query('rm_eff.notna().values')\
            .query('n_occ>20')\
            .groupby(['trinomials'])\
            .agg(
                {
                    'n_occ': {'sum', 'count'},
                    'rm_eff': 'sum', 'scaled_rm_eff': 'sum'
                }
            )
        df_tri_re['m_rm_eff'] = df_tri_re['rm_eff']['sum'] / \
            df_tri_re['n_occ']['count']
        df_tri_re = df_tri_re.reset_index()
        return rm_effect, df_tri_re

    def generate_attribution_methods(self, W):
        #  generate attention weights for all the paths
        #  analyse results - metrics, only on test_idx
        X_path_test = self.model.X_path  # [self.test_idx]
        y_test = self.dataset.y  # [self.test_idx]
        df_test = self.dataset.df_dynamic[
            ['utm_hash', 'trinomials']
            ]  # .iloc[self.test_idx]
        logger.info('generate_attribution_methods function started...')
        logger.info('Calculating sequences')
        # creating back corpus
        corpus_test = self.model.tokenizer.sequences_to_texts(X_path_test)
        all_medium = list(
            set(
                [
                    pad for lst in corpus_test
                    for pad in self.__pad_str_corpus(lst.split(' '))
                ]
            )
        )
        all_medium.remove(all_medium[0])  # remove '' from medium
        ls = [lst for lst in corpus_test]
        ls_elt = [elt.split(' ') for elt in ls]
        pad_corpus = [self.__pad_str_corpus(sublist) for sublist in ls_elt]
        logger.info('Calculating attention model weights')
        chunk_size = self.chunk_size
        logger.debug(f'Sparse chunksize: {chunk_size}')
        # chunk_size = 100000
        tc = (np.ceil(len(pad_corpus) / chunk_size).astype(int)) - 1
        smd = []  # list of sparse matrixes
        for id_chunk in range(
            np.ceil(
                len(pad_corpus) / chunk_size
                ).astype(int)
        ):
            logger.debug(f'Chunk {id_chunk}/{tc}')
            M_idc_chunk = len(
                pad_corpus[chunk_size*id_chunk:chunk_size*(id_chunk+1)]
            )
            M = np.zeros(
                (M_idc_chunk, len(pad_corpus[0]), len(all_medium)),
                dtype='float32'
            )
            for id_m, media in enumerate(all_medium):
                id_c_range = range(
                    id_chunk*chunk_size,
                    id_chunk*chunk_size+len(
                        pad_corpus[chunk_size*id_chunk:chunk_size*(id_chunk+1)]
                    )
                )
                for id_c, (id_c_fromchunk, subcorpus) in enumerate(
                    zip(
                        id_c_range,
                        pad_corpus[chunk_size*id_chunk:chunk_size*(id_chunk+1)]
                        )
                ):
                    path_len = len((corpus_test[id_c_fromchunk]))
                    if media in subcorpus:
                        att_w = (
                            (W[id_c_fromchunk, :path_len, :path_len])
                            .sum(axis=1) /
                            ((W[id_c_fromchunk, :path_len, :path_len])
                                .sum(axis=1)).sum()
                        )
                        for id_p, _ in enumerate(
                            self.__duplicates(subcorpus, media)
                        ):
                            M[id_c, id_p, id_m] = att_w[id_p]
            smd.append(sparse.COO(M))
            del M
        sM = smd[0]
        for k in range(len(smd[1:])):
            sM = sparse.concatenate((sM, smd[k]))
        mem_sM = np.round(sM.nbytes/(1024**2), 2)
        logger.debug(f'sparse matrix shape {sM.shape}')
        logger.debug(f'sparse matrix density {sM.density}')
        logger.debug(f'total sM memory usage: {mem_sM} MB')
        sM_pos = sM.sum(axis=0).T
        mem_sM_pos = np.round(sM_pos.nbytes/(1024**2), 2)
        logger.debug(f'position matrix shape {sM_pos.shape}')
        logger.debug(f'position matrix density {sM_pos.density}')
        logger.debug(f'position matrix nbytes {mem_sM_pos} MB')
        assert sM_pos.sum() != float('inf'), 'sM_pos is inf, \
            it will lead to a attention_weight = 0. \
            Try to change Mdtype - experiment.py line 150.'
        # df_utm = pd.DataFrame(
        #     (sM_pos.sum(axis=1)/sM_pos.sum()).todense(),
        #     index=all_medium, columns=['norm_att']
        # )
        df_utm = pd.DataFrame.sparse.from_spmatrix(
            (sM_pos.sum(axis=1) / sM_pos.sum()).reshape((-1, 1)),
            index=all_medium,
            columns=['norm_att']
            )
        df_utm = df_utm.sparse.to_dense()
        mem_df_utm = np.round(
            df_utm.memory_usage(
                index=True,
                deep=True).sum()/(1024**2), 2)
        logger.debug(f'df_utm memory usage: {mem_df_utm} MB')
    #     FIRST TOUCH
        logger.info('Calculating first touch model weights')
        ft_count = {
            medium: np.count_nonzero(
                X_path_test[
                    y_test == 1
                    ][:, 0] == idx
                    )
            for medium, idx
            in self.model.tokenizer.word_index.items()
        }
        df_utm['first_touch'] = pd.DataFrame.from_dict(
            ft_count, orient='index'
        )
        df_utm['first_touch'] = (
            df_utm['first_touch'] / df_utm['first_touch'].sum()
        )
    #     LAST TOUCH
        logger.info('Calculating last touch model weights')
        last_medium = [
            ((subcorpus).split(' ')[-1]).lower()
            for subcorpus in corpus_test
            ]
        lm_to_drop = self.__duplicates(last_medium, '')
        lyt = list(y_test)
        lm_to_drop.reverse()
        for null_idx in lm_to_drop:
            last_medium.pop(null_idx)
            lyt.pop(null_idx)
        lyt = np.array(lyt)
        last_idx = [self.model.tokenizer.word_index[lm] for lm in last_medium]
        last_idx = np.array(last_idx)
        lt_count = {
            medium: np.count_nonzero(
                    last_idx[lyt == 1] == idx
                )
            for medium, idx
            in self.model.tokenizer.word_index.items()
        }
        df_utm['last_touch'] = pd.DataFrame.from_dict(lt_count, orient='index')
        df_utm['last_touch'] = (
            df_utm['last_touch'] / df_utm['last_touch'].sum()
        )
    #    TOUCH POSITION BASED ATTRIBUTION
    # OLDER-> Linear: 40 - 20 - 40 attribution
    # -> NEW: - 60 - 20 - 20 from order
        logger.info('Calculating position touch model weights')
        pos_attribution_fac_first = 0.6
        pos_attribution_fac_last = 0.2
        pos_attribution_fac_rest = 1 \
            - pos_attribution_fac_last \
            - pos_attribution_fac_first
        unique_subcorpus = []
        unique_subcorpus_weight = []
        for subcorpus in corpus_test:
            subcorpus = subcorpus.split(' ')
            unique_subcorpus.append(list(set(subcorpus)))
            unique_subsub_corpus = list(set(subcorpus))
            slen = len(subcorpus)
            if slen > 2:
                fac = pos_attribution_fac_rest/(slen-2)
                subfac = []
                subfac.append(pos_attribution_fac_first)
                for _ in subcorpus[1:(slen-1)]:
                    subfac.append(fac)
                subfac.append(pos_attribution_fac_last)
                subfaclist = []
                for jj in range(len(unique_subsub_corpus)):
                    sumsubfac = 0
                    for idx in self.__duplicates(
                        subcorpus, unique_subsub_corpus[jj]
                    ):
                        sumsubfac += subfac[idx]
                    subfaclist.append(sumsubfac)
                unique_subcorpus_weight.append(subfaclist)
            elif slen == 2:
                subfac = []
                subfac.append(0.5)
                subfac.append(0.5)
                subfaclist = []
                for jj in range(len(unique_subsub_corpus)):
                    sumsubfac = 0
                    for idx in self.__duplicates(
                        subcorpus, unique_subsub_corpus[jj]
                    ):
                        sumsubfac += subfac[idx]
                    subfaclist.append(sumsubfac)
                unique_subcorpus_weight.append(subfaclist)
            else:
                unique_subcorpus_weight.append([1.0])
        flat_unique_corpus = [
            item for sublist in list(
                np.array(unique_subcorpus)[y_test == 1]
                )
            for item in sublist
        ]
        flat_unique_weights = [
            item for sublist in list(
                np.array(
                    unique_subcorpus_weight)[
                        y_test == 1
                        ]
                )
            for item in sublist
        ]
        pos_t = {}
        for medium in all_medium:
            if medium in flat_unique_corpus:
                # find the places in list that
                #  medium appears and sum all the weights,
                # then multiply by the number of times each medium appears
                pos_t_num = (
                    (np.array(
                        flat_unique_weights
                        )[
                            np.array(
                                self.__duplicates(flat_unique_corpus, medium)
                                )
                        ])
                    .sum()
                )
                pos_t[medium] = pos_t_num
            else:
                pos_t[medium] = 0.0
        df_utm['position_based'] = pd.DataFrame.from_dict(
            pos_t, orient='index'
            )
        df_utm['position_based'] = (
            df_utm['position_based'] / df_utm['position_based'].sum()
        )
    # creating exploded dataframe utm_hash and trinomials
    # and tri2hash dict
        logger.info('Calculating trinomials 2 utm_hash dictionary')
        df_exp = df_test
        exp_trinomials = df_exp['trinomials'].explode()
        df_exp = df_exp.explode('utm_hash')
        df_exp['exp_trinomials'] = exp_trinomials
        del exp_trinomials
        tri2hash = (df_exp.groupby(
            ['exp_trinomials']
            )['utm_hash']
            .apply(list)
            .apply(set)
            .apply(list)
            .to_dict())
    # creating datafame by sum of all hashs attacheced to them
        logger.info('Calculating dataframe trinomials and weights')
        df_tri = pd.DataFrame(
            index=tri2hash.keys(),
            columns=['norm_att', 'first_touch', 'last_touch', 'position_based']
            )
        for key in tri2hash.keys():
            v_att, v_ft, v_lt, v_lint = 0.0, 0.0, 0.0, 0.0
            for ii in range(len(tri2hash[key])):
                try:
                    st = df_utm[[
                        'norm_att',
                        'first_touch',
                        'last_touch',
                        'position_based'
                        ]].loc[tri2hash[key][ii]]
                    v_att = v_att+st[0]
                    v_ft = v_ft+st[1]
                    v_lt = v_lt+st[2]
                    v_lint = v_lint+st[3]
                except Exception:
                    v_att, v_ft, v_lt, v_lint = v_att, v_ft, v_lt, v_lint
            df_tri['norm_att'].loc[key] = v_att
            df_tri['first_touch'].loc[key] = v_ft
            df_tri['last_touch'].loc[key] = v_lt
            df_tri['position_based'].loc[key] = v_lint
        logger.info('Calculating dataframe for weights and positions')
        df_utmpos = pd.DataFrame.sparse.from_spmatrix(sM_pos, index=all_medium)
        pos_sum = df_utmpos.sum(axis=1)
        df_utmpos = df_utmpos.div(pos_sum.astype('float64'), axis=0)
        df_utmpos['sum'] = pos_sum

        df_tripos = df_utmpos.join(
            df_exp.groupby(['exp_trinomials', 'utm_hash'])
            .size().to_frame().reset_index()
            .set_index('utm_hash').rename(columns={0: 'cnt'})
            ).reset_index()
        df_tripos[df_tripos.columns[1:-3]] = df_tripos[
            df_tripos.columns[1:-3]].sparse.to_dense()
        df_tripos = df_tripos.groupby('exp_trinomials').sum()
        df_tripos[df_tripos.columns[:-2]] = (
            df_tripos[df_tripos.columns[:-2]].div(
                df_tripos[df_tripos.columns[:-2]].sum(axis=1), axis=0
                )
        )
        # just for the same plotting
        # function works on deving media_attrib
        df_utmpos['cnt'] = pos_sum
        logger.debug(f'df_utmpos shape {df_utmpos.shape}')
        logger.info('Attribution methods comparisons done!')
        return df_tri, df_tripos, df_utm, df_utmpos

    def run(self):
        logger.info(
            f"Begin Experiment {self.run_tag} \
            for model {self.model.model_id}"
        )
        try:
            history, _ = self.fit()
            res, att_weights, y_test, y_pred = self.evaluate_model()
            logger.info("Train Finished")
            logger.info("Generating Removal Effect")
            rm_effect, df_tri_re = self.generate_removal_effect()
            logger.info("Generating Attribution Methods")
            df_tri, df_tripos, df_utm, df_utmpos = \
                self.generate_attribution_methods(W=att_weights)
            logger.info("Comparison of attribution methods Finished")
            logger.info("Metrics for Classification:")
            results = {}
            for name, value in zip(self.model.model.metrics_names, res):
                logger.info(f'{name}:  {value}')
                results[name] = value
        except Exception as e:
            """
            Write out an error file.
            This will be returned as the failureReason
            """
            self.artifacts_handler.training_error(e)
            """
            A non-zero exit code causes the
            training job to be marked as Failed.
            """
            sys.exit(255)
# Generating figures
        figs = {}
        # metrics tensorflow
        fig_name, fig = Plotter().plot_tf_metrics(history)
        figs[fig_name] = fig
        # binary confusion matrix
        fig_name, fig = Plotter().plot_cm(
            y_test,
            y_pred[self.test_idx],
            self.hyperparameters['prob_threshold']
        )
        figs[fig_name] = fig
        #  Weight by touchpoint
        fig_name = 'Average attribution weight by touchpoints'
        fig_name, fig = Plotter().plot_weight_by_tp(
                df=self.dataset.df_dynamic,
                W=att_weights,
                fig_name=fig_name,
                path_len=self.dataset.seq_maxlen
            )
        figs[fig_name] = fig
        #  dataframes polots
        # dataframe trinomials
        fig_name = 'Tri_Attribution methods comparison'
        fig_name_pos = 'Tri_DNAM Attribution score per touch point'
        fig_name, fig, fig_name_pos, fig_pos = Plotter(). \
            plot_comparison_attribution_methods(
                df_tri, df_tripos,
                fig_name=fig_name,
                fig_name_pos=fig_name_pos,
                max_mediuns=10
            )
        figs[fig_name] = fig
        figs[fig_name_pos] = fig_pos

        fig_name = 'UTM_Attribution methods comparison'
        fig_name_pos = 'UTM_DNAM Attribution score per touch point'
        fig_name, fig, fig_name_pos, fig_pos = Plotter(). \
            plot_comparison_attribution_methods(
                df_utm, df_utmpos,
                fig_name=fig_name,
                fig_name_pos=fig_name_pos,
                max_mediuns=10
            )
        figs[fig_name] = fig
        figs[fig_name_pos] = fig_pos

        # to save CSV with 'utm_hash' or 'medium'
        # not only on index but as column
        df_tri = df_tri.reset_index() \
            .rename(columns={'index': 'trinomials'})
        df_tripos = df_tripos.reset_index() \
            .rename(columns={'exp_trinomials': 'trinomials'})
        df_utm = df_utm.reset_index() \
            .rename(columns={'index': 'utm_hash'})
        df_utmpos = df_utmpos.reset_index() \
            .rename(columns={'index': 'utm_hash'})
        df_utmpos = df_utmpos.drop(['cnt', 'sum'], axis=1).join(
            df_utm[
                ['norm_att', 'first_touch', 'last_touch', 'position_based']
            ]
            ).rename(columns={
                'norm_att': 'dl_weights'}
        )
        df_tripos = df_tripos.drop(['cnt', 'sum'], axis=1).join(
            df_tri[
                ['norm_att', 'first_touch', 'last_touch', 'position_based']
            ]
            ).rename(columns={
                'norm_att': 'dl_weights'}
        )
        mem_df = np.round(
            df_utmpos.memory_usage(
                index=True,
                deep=True).sum()/(1024**2), 2)
        logger.debug(f'Total shape of df_utmpos: {df_utmpos.shape}')
        logger.debug(f'Total memory usage of df_utmpos: {mem_df} MB')
        self.artifacts_handler.save(
            {
                "model": self.model,
                "metrics_clf": results,
                "hyperparameters": self.hyperparameters,
                "df_tri": df_tri,
                "df_tripos": df_tripos,
                "df_utm": df_utm,
                "df_utmpos": df_utmpos,
                "rm_effect": rm_effect,
                "df_tri_re": df_tri_re,
                'figures_clf': figs,
            }
        )
