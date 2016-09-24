## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

from sys import stdout as so
import os
import inspect

import d2o
import keepers

from nifty import __version__


MPI = d2o.config.dependency_injector[
        keepers.get_Configuration('D2O')['mpi_module']]

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank


class switch(object):
    """
        ..                            __   __               __
        ..                          /__/ /  /_            /  /
        ..     _______  __     __   __  /   _/  _______  /  /___
        ..   /  _____/ |  |/\/  / /  / /  /   /   ____/ /   _   |
        ..  /_____  /  |       / /  / /  /_  /  /____  /  / /  /
        .. /_______/   |__/\__/ /__/  \___/  \______/ /__/ /__/  class

        NIFTY support class for switches.

        Parameters
        ----------
        default : bool
            Default status of the switch (default: False).

        See Also
        --------
        notification : A derived class for displaying notifications.

        Examples
        --------
        >>> option = switch()
        >>> option.status
        False
        >>> option
        OFF
        >>> print(option)
        OFF
        >>> option.on()
        >>> print(option)
        ON

        Attributes
        ----------
        status : bool
            Status of the switch.

    """
    def __init__(self,default=False):
        """
            Initilizes the switch and sets the `status`

            Parameters
            ----------
            default : bool
                Default status of the switch (default: False).

        """
        self.status = bool(default)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on(self):
        """
            Switches the `status` to True.

        """
        self.status = True

    def off(self):
        """
            Switches the `status` to False.

        """
        self.status = False


    def toggle(self):
        """
            Switches the `status`.

        """
        self.status = not self.status

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        if(self.status):
            return "ON"
        else:
            return "OFF"

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class notification(switch):
    """
        ..                           __     __   ____   __                       __     __
        ..                         /  /_  /__/ /   _/ /__/                     /  /_  /__/
        ..     __ ___    ______   /   _/  __  /  /_   __   _______   ____ __  /   _/  __   ______    __ ___
        ..   /   _   | /   _   | /  /   /  / /   _/ /  / /   ____/ /   _   / /  /   /  / /   _   | /   _   |
        ..  /  / /  / /  /_/  / /  /_  /  / /  /   /  / /  /____  /  /_/  / /  /_  /  / /  /_/  / /  / /  /
        .. /__/ /__/  \______/  \___/ /__/ /__/   /__/  \______/  \______|  \___/ /__/  \______/ /__/ /__/  class

        NIFTY support class for notifications.

        Parameters
        ----------
        default : bool
            Default status of the switch (default: False).
        ccode : string
            Color code as string (default: "\033[0m"). The surrounding special
            characters are added if missing.

        Notes
        -----
        The color code is a special ANSI escape code, for a list of valid codes
        see [#]_. Multiple codes can be combined by seperating them with a
        semicolon ';'.

        References
        ----------
        .. [#] Wikipedia, `ANSI escape code <http://en.wikipedia.org/wiki/ANSI_escape_code#graphics>`_.

        Examples
        --------
        >>> note = notification()
        >>> note.status
        True
        >>> note.cprint("This is noteworthy.")
        This is noteworthy.
        >>> note.cflush("12"); note.cflush('3')
        123
        >>> note.off()
        >>> note.cprint("This is noteworthy.")
        >>>

        Raises
        ------
        TypeError
            If `ccode` is no string.

        Attributes
        ----------
        status : bool
            Status of the switch.
        ccode : string
            Color code as string.

    """
    _code = "\033[0m" ## "\033[39;49m"
    _ccode_red = "\033[31;1m"
    _ccode_yellow = "\033[33;1m"
    _ccode_green = "\033[32;1m"
    def __init__(self,default=True,ccode="\033[0m"):
        """
            Initializes the notification and sets `status` and `ccode`

            Parameters
            ----------
            default : bool
                Default status of the switch (default: False).
            ccode : string
                Color code as string (default: "\033[0m"). The surrounding
                special characters are added if missing.

            Raises
            ------
            TypeError
                If `ccode` is no string.

        """
        self.status = bool(default)

        ## check colour code
        if(not isinstance(ccode,str)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        if(ccode[0]!="\033"):
            ccode = "\033"+ccode
        if(ccode[1]!='['):
            ccode = ccode[0]+'['+ccode[1:]
        if(ccode[-1]!='m'):
            ccode = ccode+'m'
        self.ccode = ccode

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_ccode(self,newccode=None):
        """
            Resets the the `ccode` string.

            Parameters
            ----------
            newccode : string
                Color code as string (default: "\033[0m"). The surrounding
                characters "\033", '[', and 'm' are added if missing.

            Returns
            -------
            None

            Raises
            ------
            TypeError
                If `ccode` is no string.

            Examples
            --------
            >>> note = notification()
            >>> note.set_ccode("31;1") ## "31;1" corresponds to red and bright

        """
        if(newccode is None):
            newccode = self._code
        else:
            ## check colour code
            if(not isinstance(newccode,str)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            if(newccode[0]!="\033"):
                newccode = "\033"+newccode
            if(newccode[1]!='['):
                newccode = newccode[0]+'['+newccode[1:]
            if(newccode[-1]!='m'):
                newccode = newccode+'m'
        self.ccode = newccode

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _get_caller(self):
        result = ''
        i = 2
        current = inspect.stack()[i][3]
        while current != '<module>':
            result = '.' + current + result
            i += 1
            current = inspect.stack()[i][3]
        return result[1:]

    def cstring(self, subject):
        """
            Casts an object to a string and augments that with a colour code.

            Parameters
            ----------
            subject : {string, object}
                String to be augmented with a color code. A given object is
                cast to its string representation by :py:func:`str`.

            Returns
            -------
            cstring : string
                String augmented with a color code.

        """
        if rank == 0:
            return self.ccode + str(self._get_caller()) + ':\n' + \
                   str(subject) + self._code + '\n'

    def cflush(self, subject):
        """
            Flushes an object in its colour coded sting representation to the
            standard output (*without* line break).

            Parameters
            ----------
            subject : {string, object}
                String to be flushed. A given object is
                cast to a string by :py:func:`str`.

            Returns
            -------
            None

        """
        if self.status and rank == 0:
            so.write(self.cstring(subject))
            so.flush()

    def cprint(self, subject):
        """
            Flushes an object in its colour coded sting representation to the
            standard output (*with* line break).

            Parameters
            ----------
            subject : {string, object}
                String to be flushed. A given object is
                cast to a string by :py:func:`str`.

            Returns
            -------
            None

        """
        if self.status and rank == 0:
            so.write(self.cstring(subject)+"\n")
            so.flush()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        if(self.status):
            return self.cstring("ON")
        else:
            return self.cstring("OFF")

##-----------------------------------------------------------------------------



class _about(object): ## nifty support class for global settings
    """
        NIFTY support class for global settings.

        .. warning::
            Turning off the `_error` notification will suppress all NIFTY error
            strings (not recommended).

        Examples
        --------
        >>> from nifty import *
        >>> about
        nifty version 0.2.0
        >>> print(about)
        nifty version 0.2.0
        - errors          = ON (immutable)
        - warnings        = ON
        - infos           = OFF
        - multiprocessing = ON
        - hermitianize    = ON
        - lm2gl           = ON
        >>> about.infos.on()
        >>> about.about.save_config()

        >>> from nifty import *
        INFO: nifty version 0.2.0
        >>> print(about)
        nifty version 0.2.0
        - errors          = ON (immutable)
        - warnings        = ON
        - infos           = ON
        - multiprocessing = ON
        - hermitianize    = ON
        - lm2gl           = ON

        Attributes
        ----------
        warnings : notification
            Notification instance controlling whether warings shall be printed.
        infos : notification
            Notification instance controlling whether information shall be
            printed.
        multiprocessing : switch
            Switch instance controlling whether multiprocessing might be
            performed.
        hermitianize : switch
            Switch instance controlling whether hermitian symmetry for certain
            :py:class:`rg_space` instances is inforced.
        lm2gl : switch
            Switch instance controlling whether default target of a
            :py:class:`lm_space` instance is a :py:class:`gl_space` or a
            :py:class:`hp_space` instance.

    """
    def __init__(self):
        """
            Initializes the _about and sets the attributes.

        """
        ## version
        self._version = str(__version__)

        ## switches and notifications
        self._errors = notification(default=True,
                                    ccode=notification._code)
        self.warnings = notification(default=True,
                                     ccode=notification._code)
        self.infos =  notification(default=False,
                                   ccode=notification._code)
        self.multiprocessing = switch(default=True)
        self.hermitianize = switch(default=True)
        self.lm2gl = switch(default=True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def load_config(self,force=True):
        """
            Reads the configuration file "~/.nifty/nifty_config".

            Parameters
            ----------
            force : bool
                Whether to cause an error if the file does not exsist or not.

            Returns
            -------
            None

            Raises
            ------
            ValueError
                If the configuration file is malformed.
            OSError
                If the configuration file does not exist.

        """
        nconfig = os.path.expanduser('~')+"/.nifty/nifty_config"
        if(os.path.isfile(nconfig)):
            rawconfig = []
            with open(nconfig,'r') as configfile:
                for ll in configfile:
                    if(not ll.startswith('#')):
                        rawconfig += ll.split()
            try:
                self._errors = notification(default=True,ccode=rawconfig[0])
                self.warnings = notification(default=int(rawconfig[1]),ccode=rawconfig[2])
                self.infos =  notification(default=int(rawconfig[3]),ccode=rawconfig[4])
                self.multiprocessing = switch(default=int(rawconfig[5]))
                self.hermitianize = switch(default=int(rawconfig[6]))
                self.lm2gl = switch(default=int(rawconfig[7]))
            except(IndexError):
                raise ValueError(about._errors.cstring("ERROR: '"+nconfig+"' damaged."))
        elif(force):
            raise OSError(about._errors.cstring("ERROR: '"+nconfig+"' nonexisting."))

    def save_config(self):
        """
            Writes to the configuration file "~/.nifty/nifty_config".

            Returns
            -------
            None

        """
        rawconfig = [self._errors.ccode[2:-1],str(int(self.warnings.status)),self.warnings.ccode[2:-1],str(int(self.infos.status)),self.infos.ccode[2:-1],str(int(self.multiprocessing.status)),str(int(self.hermitianize.status)),str(int(self.lm2gl.status))]

        nconfig = os.path.expanduser('~')+"/.nifty/nifty_config"
        if(os.path.isfile(nconfig)):
            rawconfig = [self._errors.ccode[2:-1],str(int(self.warnings.status)),self.warnings.ccode[2:-1],str(int(self.infos.status)),self.infos.ccode[2:-1],str(int(self.multiprocessing.status)),str(int(self.hermitianize.status)),str(int(self.lm2gl.status))]
            nconfig = os.path.expanduser('~')+"/.nifty/nifty_config"

            with open(nconfig,'r') as sourcefile:
                with open(nconfig+"_",'w') as targetfile:
                    for ll in sourcefile:
                        if(ll.startswith('#')):
                            targetfile.write(ll)
                        else:
                            ll = ll.replace(ll.split()[0],rawconfig[0]) ## one(!) per line
                            rawconfig = rawconfig[1:]
                            targetfile.write(ll)
            os.rename(nconfig+"_",nconfig) ## overwrite old congiguration
        else:
            if(not os.path.exists(os.path.expanduser('~')+"/.nifty")):
                os.makedirs(os.path.expanduser('~')+"/.nifty")
            with open(nconfig,'w') as targetfile:
                for rr in rawconfig:
                    targetfile.write(rr+"\n") ## one(!) per line

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "nifty version "+self._version

    def __str__(self):
        return "nifty version "+self._version+"\n- errors          = "+self._errors.cstring("ON")+" (immutable)\n- warnings        = "+str(self.warnings)+"\n- infos           = "+str(self.infos)+"\n- multiprocessing = "+str(self.multiprocessing)+"\n- hermitianize    = "+str(self.hermitianize)+"\n- lm2gl           = "+str(self.lm2gl)

##-----------------------------------------------------------------------------

## set global instance
about = _about()
#about.load_config(force=False)
#about.infos.cprint("INFO: "+about.__repr__())


