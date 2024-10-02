# -*- coding: utf-8 -*-
'''
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present package offers a tool, to support the user in the task of data analysis of multiple aspect trajectories. It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

Authors:
    - Tarlis Portela
    - Lucas May Petry (adapted)
'''
# --------------------------------------------------------------------------------
import sys
from datetime import datetime

class Logger(object):
    """
    A simple logging class to log messages with different severity levels.
    """

    LOG_LINE = None
    INFO        = '[    INFO    ]'
    WARNING     = '[  WARNING   ]'
    ERROR       = '[   ERROR    ]'
    CONFIG      = '[   CONFIG   ]'
    RUNNING     = '[  RUNNING   ]'
    QUESTION    = '[  QUESTION  ]'

    def log(self, type, message):
        """
        Log a message with a specific type.

        Parameters:
        -----------
        log_type : str
            The type of the log (e.g., INFO, WARNING, ERROR).
        
        message : str
            The message to log.
        """
        if Logger.LOG_LINE:
            sys.stdout.write("\n")
            sys.stdout.flush()
            Logger.LOG_LINE = None

        sys.stdout.write(str(type) + " " + self.cur_date_time() + " :: " + message + "\n")
        sys.stdout.flush()

    def log_dyn(self, type, message):
        """
        Log a message dynamically (overwrites the same line).

        Parameters:
        -----------
        log_type : str
            The type of the log (e.g., INFO, WARNING, ERROR).
        
        message : str
            The message to log.
        """
        line = str(type) + " " + self.cur_date_time() + " :: " + message
        sys.stdout.write("\r\x1b[[" + line.__str__())
        sys.stdout.flush()
        Logger.LOG_LINE = line

    def get_answer(self, message):
        """
        Get user input with a prefixed question log.

        Parameters:
        -----------
        message : str
            The question to prompt the user.
        
        Returns:
        --------
        str
            The user input.
        """
        return input(Logger.QUESTION + " " + self.cur_date_time() + " :: " + message)

    def cur_date_time(self):
        """
        Get the current date and time in a formatted string.

        Returns:
        --------
        str
            The current date and time as a string.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")