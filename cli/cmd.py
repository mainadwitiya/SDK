import argparse
import copy
import datetime
from functools import wraps
import getpass
import json
import logging
import os
from os import path
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import xtarfile as tarfile
import click
from click.exceptions import ClickException
import six
from six.moves import configparser
import yaml
import alectio_sdk
import requests
from alectio_sdk.cli.alectio_cli import AlectioClient
from alectio_sdk.cli.datasets.downloader import Downloader
from tabulate import tabulate
import urllib.request
from os.path import expanduser
import os
import yaml


LOG_STRING = click.style("alectio", fg="blue", bold=True)
LOG_STRING_NOCOLOR = "alectio"
ERROR_STRING = click.style("ERROR", bg="red", fg="green")
WARN_STRING = click.style("WARNING", fg="yellow")
PRINTED_MESSAGES = set()

_silent = False
_show_info = True
_show_warnings = True
_show_errors = True
_logger = None

CONTEXT = dict(default_map={})

downloader = Downloader('universal')
def termerror(string, **kwargs):
    string = "\n".join(["{} {}".format(ERROR_STRING, s) for s in string.split("\n")])
    _log(
        string=string,
        newline=True,
        silent=not _show_errors,
        level=logging.ERROR,
        **kwargs
    )


def termwarn(string, **kwargs):
    string = "\n".join(["{} {}".format(WARN_STRING, s) for s in string.split("\n")])
    _log(
        string=string,
        newline=True,
        silent=not _show_errors,
        level=logging.WARN,
        **kwargs
    )

def _log(
    string="", newline=True, repeat=True, prefix=True, silent=False, level=logging.INFO
):
    global _logger
    silent = silent or _silent
    if string:
        if prefix:
            line = "\n".join(
                ["{}: {}".format(LOG_STRING, s) for s in string.split("\n")]
            )
        else:
            line = string
    else:
        line = ""
    if not repeat and line in PRINTED_MESSAGES:
        return
    # Repeated line tracking limited to 1k messages
    if len(PRINTED_MESSAGES) < 1000:
        PRINTED_MESSAGES.add(line)
    if silent:
        if level == logging.ERROR:
            logging.error(line)
        elif level == logging.WARNING:
            logging.warning(line)
        else:
            logging.info(line)
    else:
        click.echo(line, file=sys.stderr, nl=newline)

def cli_unsupported(argument):
    termerror("Unsupported argument `{}`".format(argument))
    sys.exit(1)


# class ClickWandbException(ClickException):
#     def format_message(self):
#         # log_file = util.get_log_file_path()
#         log_file = ""
#         orig_type = "{}.{}".format(self.orig_type.__module__, self.orig_type.__name__)
#         if issubclass(self.orig_type, Error):
#             return click.style(str(self.message), fg="red")
#         else:
#             return "An Exception was raised, see %s for full traceback.\n" "%s: %s" % (
#                 log_file,
#                 orig_type,
#                 self.message,
#             )

def display_error(func):
    """Function decorator for catching common errors and re-raising as wandb.Error"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            logging.error("".join(lines))

    return wrapper
class RunGroup(click.Group):
    @display_error
    def get_command(self, ctx, cmd_name):
        # TODO: check if cmd_name is a file in the current dir and not require `run`?
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        return None


def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

## Click commands

@click.group()
@click.version_option(version=alectio_sdk.__version__)
@click.pass_context
def main(ctx):
    pass
    # if ctx.invoked_subcommand is None:
    #     click.echo(ctx.get_help())

@main.command(name='login')
@click.argument("key", nargs=-1)
@click.option(
    "--relogin", default=None, is_flag=True, help="Force relogin if already logged in."
)
@click.option("--anonymously", default=False, is_flag=True, help="Log in anonymously")
def login(key, relogin, anonymously, no_offline=False):

    if len(key) == 0:
        termerror("API Key is required.")
        sys.exit(0)

    key = key[0]
    
    headers = {"Authorization": "Bearer " + key}
    response = requests.post('https://api.alectio.com/api/me', headers=headers)
    if response.status_code == 401:
        termerror("API Key has expired or malformed")
    elif response.status_code == 200:
        payload = response.json()
        payload['api_key'] = key
        payload['user_id'] = payload['id']
        dict_file = [payload]

        alectio_dir = expanduser("~/.alectio")
        credential_file = expanduser(alectio_dir+'/credentials.yaml')
        if os.path.isdir(alectio_dir) and os.path.isfile(credential_file):
            termerror("Auth data already exists. If you want to reinitialize run alectio login with --relogin flag.")
        else:
            os.mkdir(alectio_dir) if not os.path.isdir(alectio_dir) else _log('Alectio dir exists')
            try:
                with open(credential_file, 'w') as file:
                    yaml.dump(dict_file, file)
            except:
                termerror('Not able to wirte file at ' +  credential_file)
            _log('Alectio Login complete')
    else:
        termerror("Something went wrong double check you API Key")

    _log('You are logged in as: ' + payload['username'])



@main.command(name='get')
@click.argument("entity", nargs=-1)
@click.option("--project", default=None, help="Specify project id, not sure of project id run 'alectio get projects'")
def get(entity, project):
    entity = entity[0]
    alectio = AlectioClient()
    if entity == 'projects':
        projects = alectio.projects()
        projects_table = []

        for project in projects:
            projects_table.append([project._name, project._id])        
        print(tabulate(projects_table, headers=['Name', 'Project ID'], tablefmt='fancy_grid'))
    
    if entity == 'project':
        if project is None:
            termerror("Project argument use 'alectio get project --project=PROJECT_ID'")
            sys.exit(1)
        else:
            project_data = alectio.project(project)
            print(project_data._name)
            print(project_data._attr['type'])
            print(project_data._id)
            
            experiments = project_data.experiments()
            for experiment in experiments:
                print(experiment._attr)
                print(experiment._name)
                print(experiment._id)


    if entity == 'experiments':
        experiments_table = []
        if project is None:
            termerror("Project argument use 'alectio get experiments --project=PROJECT_ID'")
            sys.exit(1)
        else:
            experiments = alectio.experiments(project)
            for experiment in experiments:
                experiments_table.append([experiment._name, experiment._id])        
                print(tabulate(experiments_table, headers=['Name', 'Experiment ID'], tablefmt='fancy_grid'))

    if entity == 'experiment':
        pass


@main.command(name='library-run')
@click.option("--token", default=None, help="Enter the Token you after creating an experiment for alectio library project")
@click.option("--resume", is_flag=True, default=False, help="Enter the Token you after creating an experiment for alectio library project")
@click.option("--auto_run", is_flag=True, default=False, help="Enter the Token you after creating an experiment for alectio library project")
def library_run(token, resume, auto_run):
    if path.exists('./.alectio') and not resume:
        termwarn('An experiment initialized here. Use either alectio library-run --resume or start a new experiment with alectio clear')
        sys.exit(0)
    elif path.exists('./.alectio') and resume:
        pass

    else:
        if token is None:
            termerror("Token is required.")
            sys.exit(1)
        _log("Experiment Initilizing.")

        if not resume:
            payload = requests.post('https://api.alectio.com/experiments/libraryRunFetch', json={'token': token}).json()
        else:
            payload_file = open('./.alectio/alectio_env.json')
            payload = json.load(payload_file)
            payload_file.close()

        if payload['status'] == 'error':
            termerror(payload['message'])
            sys.exit(1)
        
        if payload['data_url'] is None or payload['code_url'] is None:
            termerror('This Dataset or model is not supported it. Please check back')
            sys.exit(1)

        else:
            if not path.exists('./.alectio'):
                os.mkdir('./.alectio')
                with open('./.alectio/alectio_env.json', 'w') as fp:
                    json.dump(payload, fp)
            
            while True:
                if path.exists('./'+payload['code_file']):
                    break
                try:
                    urllib.request.urlretrieve(payload['code_url'], payload['code_file'])
                except Exception:
                    urllib.request.urlretrieve(payload['code_url'], payload['code_file'])
                else:
                    break


            # while True:

            #     response = requests.get(payload['code_url'], verify=False, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}, stream=True)
            #     if response.status_code == 200:
            #         with open(payload['code_file'], 'wb') as f:
            #             f.write(response.raw.read())
            #     else:
            #         if download == 2:
            #             termerror('Unable to get model. Please contact info@alectio.com if this persists')
            #             sys.exit(1)
            #         else:
            #             termerror('Retrying to fetch the model')
            #     download += 1
            if not path.exists('./'+payload['code_file'].replace('.tar.gz', "")):
                with tarfile.open(payload['code_file'], 'r') as archive:
                    archive.extractall()

            os.chdir(payload['code_file'].replace('.tar.gz', ''))
            if not path.exists('./data'):
                os.mkdir('data')
    
            if not path.exists('./log'):
                os.mkdir('log')

            if not path.exists('./weights'):
                os.mkdir('weights')
            
            if not path.exists('./weight'):
                os.mkdir('weight')

            if payload['data_url'] == 'Inplace':
                _log('Data will be dowloaded on first run of code')
        
            elif payload['data_url'] == 'Internal':
                pass

            else:
                while True:
                    if path.exists('./data/'+payload['data_file']):
                        break
                    try:
                        urllib.request.urlretrieve(payload['data_url'], "./data/"+payload['data_file'])
                    except Exception:
                        urllib.request.urlretrieve(payload['data_url'], "./data/"+payload['data_file'])
                    else:
                        break
                # downloader.download_file(payload['data_url'], './data' + payload['data_file'])
                _log('Extracting Data into data dir')
                os.chdir('./data')
                if not path.exists('./'+payload['data_file'].replace('.tar.gz', "")):
                    with tarfile.open('./' + payload['data_file'], 'r') as archive:
                        archive.extractall()
                os.chdir('../')

                _log('All files are feteched.')

                if auto_run:
                    pass
                else:
                    print('Run you experiment by installing all dependencies with %s/requirements.txt and the run running python main.py %s at: %s'  % (os.getcwd(), token, os.getcwd()))




        



