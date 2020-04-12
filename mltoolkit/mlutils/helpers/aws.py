from plumbum import local


def execute_cli(*args):
    """
    Executes a CLI command, where the first args is the command, and others are
    parameters.
    """
    executable = args[0]
    params = args[1:]
    cmd = local[executable][params]
    return cmd()


def aws_s3_ls(path, recursive=False):
    """Wrapper around the 'aws s3 ls [path]' CLI command """
    cmd = ['aws', 's3', 'ls', path]
    if recursive:
        cmd += ['--recursive']
    return execute_cli(*cmd)
