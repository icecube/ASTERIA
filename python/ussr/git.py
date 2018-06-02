# Git interaction code adapted from https://github.com/desihub/desiutil.
# The desiutil project is distributed under a 3-clause BSD style license:
#
# Copyright (c) 2014-2017, DESI Collaboration <desi-data@desi.lbl.gov>
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the DESI Collaboration nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
"""Some code for interacting with git.
"""


def version(git='git'):
    """Use ``git describe`` to generate a version string.

    Parameters
    ----------
    git : :class:`str`, optional
        Path to the git executable, if not in :envvar:`PATH`.

    Returns
    -------
    :class:`str`
        A :pep:`386`-compatible version string.

    Notes
    -----
    The version string should be compatible with :pep:`386` and
    :pep:`440`.
    """
    from subprocess import Popen, PIPE
    myversion = '0.0.1.dev0'
    try:
        p = Popen([git, "describe", "--tags", "--dirty", "--always"],
                  universal_newlines=True, stdout=PIPE, stderr=PIPE)
    except OSError:
        return myversion
    out, err = p.communicate()
    if p.returncode != 0:
        return myversion
    ver = out.rstrip().split('-')[0]+'.dev'
    try:
        p = Popen([git, "rev-list", "--count", "HEAD"],
                  universal_newlines=True, stdout=PIPE, stderr=PIPE)
    except OSError:
        return myversion
    out, err = p.communicate()
    if p.returncode != 0:
        return myversion
    ver += out.rstrip()
    return ver
