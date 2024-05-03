# customer_segmentation
PoC repo for [RAD-111](https://dafiti.jira.com/browse/RAD-111)


## How to use:

You will need to create a file `test_dir/input/config/secrets.json` with the following content:

```
{ 
    "host":                     "<host>", 
    "dbname":                   "<dbname>",
    "port":                     "<port>",
    "user":                     "<user>",
    "password":                 "<password>",
    "bucket":                   "<bucket>",
    "prefix":                   "<prefix>", # ending with '/'
    "iam_role":                 "<iam_role>",
    "aws_access_key_id" :       "<aws_access_key_id>",
    "aws_secret_access_key":    "<aws_secret_access_key>",
    "aws_session_token":        "<aws_session_token>"
}
```
