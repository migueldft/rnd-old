# R&D Dataset Registry

This is an semi-automated repository generate for dataset challenges versioning. Please report any issues.

*Dataset Registry* is a centralized place to manage raw data files for use in other R&D projects, such as ___#TODO.

## Installation

Start by cloning the project:

```
$ git clone https://github.com/iterative/rnd_data_registry
$ cd rnd_data_registry
```

This project comes with a preconfigured DVC [remote storage](https://dvc.org/doc/command-reference/remote) to hold all of the datasets. This is a read-only HTTP remote.

```
$ dvc remote list
storage	s3://dafiti-dataset-registry
```

> To be able to push to the default remote requires having configured corresponding S3 credentials locally.

## Testing data synchronization locally

If you'd like to test commands like [`dvc push`](https://man.dvc.org/push),
that require write access to the remote storage, the easiest way would be to set
up a "local remote" on your file system:

> This kind of remote is located in the local file system, but is external to
> the DVC project.

```console
$ mkdir -P /tmp/dvc-storage
$ dvc remote add local /tmp/dvc-storage
```

You should now be able to run:

```console
$ dvc push -r local
```

## Datasets

The folder structure of this project groups datasets corresponding to the
external projects they pertain to.

#TODO how to download datasets

#TODO how to use in a CICD for experimentation.
