# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:42:22 2020

@author: Simon.Hermes
"""

import concurrent.futures
import time
import sys

import pandas as pd

from optimizationModel import OptSys
from datetime import datetime
from logManager import LogManager

logManager = LogManager()
logger_main = logManager.setup_logger()
logger_run =  logManager.setup_logger(name='logger_run', log_file='log_run.txt', level=None)


def optimization_coordinator(n_parallel=1):
    start = datetime.now()
    max_workers = 20

    if n_parallel > max_workers:
        n_parallel = max_workers
        logger_main.info('optimization_coordinator: reduce parallel processes to max_workers')

    logger_run.info('optimization_coordinator: Starting {} processes'.format(n_parallel))

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i  in range(n_parallel):
            # Start optimization instance and wait to avoid overlapping
            results.append(executor.submit(optimization_instance, i))
            time.sleep(5)

        for f in concurrent.futures.as_completed(results):
            logger_main.info('Parallel process terminated: {}'.format(f.result()))

    dt = datetime.now() - start
    logger_run.info('optimization_coordinator: Finished all processes after {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))


def optimization_instance(name):

    # Read scenario matrix of format:
    #   # Scenario matrix
    #   Name	Status
    #   sce_1	pending
    #   sce_2	running
    #   sce_3	finished
    file_name = 'scenario_matrix.txt'
    table_header = '# Scenario matrix (pending, running, finished, error)\n'

    while True:

        # Read next scenario
        logger_main.info('optimization_instance_{}: Read scenario matrix'.format(name))
        scenario_matrix = pd.read_csv(file_name, sep='\t', skiprows=1)
        sce_pending = scenario_matrix.loc[scenario_matrix.loc[:,'Status'] == 'pending', 'Name'].values
        if len(sce_pending) == 0:
            logger_main.info('optimization_instance_{}: No pending scenario'.format(name))
            break
        else:
            sce = sce_pending[0]

        # Update scenario -> running
        logger_main.info('optimization_instance_{}: Update scenario matrix'.format(name))
        scenario_matrix.loc[scenario_matrix.loc[:,'Name'] == sce, 'Status'] = 'running'

        f = open(file_name, 'w')
        f.write(table_header)
        scenario_matrix.to_csv(f, sep='\t', index=False, line_terminator='\n')
        f.close()

        # Start optimization of scenario
        logger_run.info('optimization_instance_{}: Start optimization of scenario {}'.format(name, sce))
        res = OptSys.optimize(scenario=sce)


# Update text fiel automatically in notepad++: https://www.raymond.cc/blog/monitor-log-or-text-file-changes-in-real-time-with-notepad/

        # Update scenario -> finished
        logger_main.info('optimization_instance_{}: Update scenario matrix'.format(name, sce))
        scenario_matrix = pd.read_csv(file_name, sep='\t', skiprows=1)
        scenario_matrix.loc[scenario_matrix.loc[:,'Name'] == sce, 'Status'] = 'finished' if res == 1 else 'error'
        f = open(file_name, 'w')
        f.write(table_header)
        scenario_matrix.to_csv(f, sep='\t', index=False, line_terminator='\n')
        f.close()

    logger_main.info('optimization_instance_{}: Closed optimization instance'.format(name))

    return 'optimization_instance_{}'.format(name)


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) > 1:
        n_parallel = int(arg[1])
    else:
        n_parallel = 1

    optimization_coordinator(n_parallel)


"""
NOTES

# optimization_coordinator
#def optimization_coordinator(n_parallel=1):
#    start = datetime.now()
#    logging.info('optimization_coordinator: Starting {} processes'.format(n_parallel))
#    with concurrent.futures.ProcessPoolExecutor() as executor:
#        results = [executor.submit(optimization_instance, i) for i in range(n_parallel)]
#        for f in concurrent.futures.as_completed(results):
#            logging.info('Parallel process terminated: {}'.format(f.result()))
#
#    dt = datetime.now() - start
#    logging.info('optimization_coordinator: Finished all processes after {}min {}sec'.format(int(dt.total_seconds()/60), round(dt.total_seconds()%60)))

#def main():
#	n_parallel = 2
#	optimization_coordinator(n_parallel)


"""