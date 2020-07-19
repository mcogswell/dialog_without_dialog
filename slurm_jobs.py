#!/usr/bin/env python
import subprocess
import glob
import re
import os.path as pth
import json
import time
from json.decoder import JSONDecodeError

def main():
    '''
    Display slurm jobs logged in `data/log/`. By default only running
    jobs are reported. Times indicate when the job started (log file creation).

    Usage:
        slurm_jobs.py [<id>|--last L] [--refresh]

    Options:
        -l --last L  Show the last L jobs
        -r --refresh  Delete and re-cache the list of job files
    '''
    import docopt, textwrap
    args = docopt.docopt(textwrap.dedent(main.__doc__))
    if args['<id>']:
        jid = args['<id>']
    else:
        jid = None
    log_dir = 'logs/run_logs'

    # figure out which jobs to show (currently running slurm jobs)
    show_job_ids = []
    if jid is None:
        output = subprocess.check_output('squeue -hu $USER -o %A', shell=True)
        if output:
            show_job_ids = list(output.decode().strip().split('\n'))
    else:
        show_job_ids = [jid]

    # show jobs
    job_logs = []
    job_exp_codes = []
    job_commands = []
    job_log_ids = []
    job_times = []
    log_fnames = glob.glob(pth.join(log_dir, '*.log'))
    if not args['--refresh'] and pth.exists(pth.join(log_dir, 'job_log_cache.json')):
        try:
            with open(pth.join(log_dir, 'job_log_cache.json'), 'r') as f:
                job_id_fname_list = json.load(f)['job_id_fname_list']
        except JSONDecodeError:
            # if corrupt, re-create the cache from scratch
            job_id_fname_list = []
    else:
        job_id_fname_list = []
    job_id_to_fname = {jid: f for jid, f in job_id_fname_list}
    fname_to_job_id = {f: jid for jid, f in job_id_fname_list}
    if args['--last']:
        log_fnames.sort(key=pth.getctime, reverse=True)
        log_fnames = log_fnames[:int(args['--last'])]
    nread = 0
    for fname in log_fnames:
        # only open logs whose job ids are unknown or need to be shown
        if fname in fname_to_job_id:
            job_id = fname_to_job_id[fname]
            if job_id not in show_job_ids:
                continue
        job_id = None
        nread += 1
        if nread % 500 == 0:
            if nread == 500:
                print('taking a while...')
            print('\tlog {} / {} (max)'.format(nread, len(log_fnames)))
        with open(fname, 'r') as f:
            lines = iter(f)
            try:
                line = next(lines)
            except StopIteration:
                print('log has no first line, skipping... ({})'.format(fname))
                continue
            # NOTE: Ids for jobs from job arrays use the actual job id (%A),
            # not the job array id followed by array id (%i).
            match = re.match('slurm job id: (\d+)', line)
            if match:
                job_id = match.group(1)
                if job_id in show_job_ids or args['--last']:
                    job_logs.append(fname)
                    job_log_ids.append(job_id)
                    nl = next(lines)
                    if nl.startswith('experiment: '):
                        exp_code = re.match('experiment: (.*)$', nl).group(1)
                        job_exp_codes.append(exp_code)
                        job_commands.append(next(lines))
                    else:
                        job_exp_codes.append('NA')
                        job_commands.append(nl)
                    job_times.append(pth.getctime(fname))
        if fname not in fname_to_job_id:
            job_id_fname_list.append((job_id, fname))
        else:
            assert job_id == fname_to_job_id[fname], ('The parsed job_id is '
                                        'different from the one in the cache')

    with open(pth.join(log_dir, 'job_log_cache.json'), 'w') as f:
        json.dump({'job_id_fname_list': job_id_fname_list}, f)

    for jid, log, exp, cmd, jtime in zip(job_log_ids, job_logs, job_exp_codes,
                                         job_commands, job_times):
        print(jid)
        when = time.strftime("%m/%d/%y %H:%M:%S", time.gmtime(jtime))
        print(log + '\t({})'.format(when))
        print(exp)
        print(cmd)


if __name__ == '__main__':
    main()

