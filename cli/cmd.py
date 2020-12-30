import argparse
import copy
import datetime
from functools import wraps
import getpass
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
import traceback

import click
from click.exceptions import ClickException
import six
from six.moves import configparser
import yaml
import alectio_sdk
import requests
from alectio_sdk.cli.alectio_cli import AlectioClient
from tabulate import tabulate



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

alectio = AlectioClient()
def termerror(string, **kwargs):
    string = "\n".join(["{} {}".format(ERROR_STRING, s) for s in string.split("\n")])
    _log(
        string=string,
        newline=True,
        silent=not _show_errors,
        level=logging.ERROR,
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
@click.option("--cloud", is_flag=True, help="Login to the cloud instead of local")
@click.option("--host", default=None, help="Login to a specific instance of W&B")
@click.option(
    "--relogin", default=None, is_flag=True, help="Force relogin if already logged in."
)
@click.option("--anonymously", default=False, is_flag=True, help="Log in anonymously")

def login(key, host, cloud, relogin, anonymously, no_offline=False):
    print(key)


@main.command(name='get')
@click.argument("entity", nargs=-1)
@click.option("--project", default=None, help="Specify project id, not sure of project id run 'alectio get projects'")
def get(entity, project):
    entity = entity[0]
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







