from os.path import expanduser
import os
import yaml
def extract_id(attr):
    """
    extract id from the attribute
    :params: attr primary key or sort key  
    """
    id = "".join(attr.split("#")[-1])
    return id


def credentials_exists():
    alectio_dir = expanduser("~/.alectio")
    credential_file = expanduser(alectio_dir+'/credentials.yaml')
    credential_data = None
    if os.path.isdir(alectio_dir) and os.path.isfile(credential_file):
        with open(credential_file, 'r') as stream:
            try:
                credential_data = yaml.safe_load(stream)[0]
            except yaml.YAMLError as exc:
                print(exc)
                print('\n Alectio Error: credentials.yaml is malformed\n')
                exit(1)
    os.environ['ALECTIO_API_KEY'] = credential_data['api_key']
    return True

def get_credentials():
    alectio_dir = expanduser("~/.alectio")
    credential_file = expanduser(alectio_dir+'/credentials.yaml')
    credential_data = None
    if os.path.isdir(alectio_dir) and os.path.isfile(credential_file):
        with open(credential_file, 'r') as stream:
            try:
                credential_data = yaml.safe_load(stream)[0]
                os.environ['ALECTIO_API_KEY'] = credential_data['api_key']
            except yaml.YAMLError as exc:
                print(exc)
                print('\n Alectio Error: credentials.yaml is malformed\n')
                exit(1)
    else:
        print('\n Alectio Error: credentials.yaml not found. Run alectio login YOUR_API_KEY\n')
        exit(1)
    return credential_data


