from sys import argv


execname = argv[0]

infoenabled = True
warnenabled = True
errenabled = True


def info(*msgs):
    """
    Log informative messages.

    This function logs the given objects to stdout. It prepends each object with the
    executable name (``execname``), appends a newline, converts the objects to strings
    and prints them. The logging does not happen if ``infoenabled`` is not True.

    Parameters
    ----------
    *msgs : object
        Messages to log.

    Returns
    -------
    None

    See Also
    --------
    warn : Log warning messages.
    err : Log error messages.

    Examples
    --------
    >>> info("informative message")
    execname: informative message
    >>> info("informative message 1", "informative message 2")
    execname: informative message 1
    execname: informative message 2
    """
    if not infoenabled:
        return

    for msg in msgs:
        print(f"{execname}: {msg}")


def warn(*msgs):
    """
    Log warning messages.

    This function logs the given objects to stdout. It prepends each object with the
    executable name (``execname``), appends a newline, converts the objects to strings
    and prints them. The logging does not happen if ``warnenabled`` is not True.

    Parameters
    ----------
    *msgs : object
        Messages to log.

    Returns
    -------
    None

    See Also
    --------
    info : Log informative messages.
    err : Log error messages.

    Examples
    --------
    >>> info("warning message")
    execname: warning message
    >>> info("warning message 1", "warning message 2")
    execname: warning message 1
    execname: warning message 2
    """
    if not warnenabled:
        return

    for msg in msgs:
        print(f"{execname}: {msg}")


def err(*msgs):
    """
    Log error messages.

    This function logs the given objects to stdout. It prepends each object with the
    executable name (``execname``), appends a newline, converts the objects to strings
    and prints them. The logging does not happen if ``errenabled`` is not True.

    Parameters
    ----------
    *msgs : object
        Messages to log.

    Returns
    -------
    None

    See Also
    --------
    info : Log informative messages.
    warn : Log warning messages.

    Examples
    --------
    >>> info("error message")
    execname: error message
    >>> info("error message 1", "error message 2")
    execname: error message 1
    execname: error message 2
    """
    if not errenabled:
        return

    for msg in msgs:
        print(f"{execname}: {msg}")
