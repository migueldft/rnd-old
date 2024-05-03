import contextlib

import psycopg2

from .logger import set_up_logging


logger = set_up_logging(__name__)


class RedshiftAdapter:
    """Class responsible for running queries against Redshift.
    :type host: str
    :param host: Host where RedShift is located.
    :type dbname: str
    :param dbname: name of database to connect to.
    :type port: int
    :param port: port used to establish connection.
    :type user: str
    :param user: user responsible for the session.
    :type password: str
    :param password: password to establish connection.
    """

    def __init__(self, **kwargs):
        self._con_setup = kwargs
        self._con = None

    def query(self, query):
        """Runs `query` against RedShift.
        :type query: str
        :param query: query to run.
        :rtype: list of dicts
        :returns: list  with results from Redshift operation where each key
                  is a name of a column in result set and value is
                  correspondent value. Returns empyt list if result set is
                  empty.
        """
        with self.cursor() as c:
            try:
                c.execute(query)
            except Exception as e:
                logger.error(e)
                exit(1)
            return c.fetchall() if c.description is not None else []
    
    @contextlib.contextmanager
    def con(self):
        """Context-Managed point of entrance for building connector with
        Redshift.
        """
        if not self._con:
            self._con = psycopg2.connect(**self._con_setup)
        yield self
        self._con.close()
        self._con = None

    @contextlib.contextmanager
    def cursor(self):
        """Users the connector to build a cursor that can be used to run
        queries against Redshift."""
        if not self._con:
            raise ValueError(
                'Please first initiate the connection to Redshift.')
        cur = self._con.cursor()
        yield cur
        cur.close()

    def init_con(self):
        if not self._con:
            self._con = psycopg2.connect(**self._con_setup)

    def close_con(self):
        if not self._con.closed:
            self._con.close()


def get_redshift_adapter(hparams: dict) -> None:
    required_fields = ["host", "dbname", "port", "user", "password"]
    for req in required_fields:
        if not hparams.get(req, ""):
            raise Exception(f'Param "{req}" is not defined and needed for psycopg2 connection')
    return RedshiftAdapter(
        host=hparams.get("host"),
        dbname=hparams.get("dbname"),
        port=hparams.get("port"),
        user=hparams.get("user"),
        password=hparams.get("password")
    )