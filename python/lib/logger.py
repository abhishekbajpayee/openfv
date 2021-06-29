"""A simple Google-style logging wrapper taken from the glog package."""
"""Use %%% to extend your previous log message (instead of logging a new message)"""

import logging
import time
import traceback
import os

import gflags as flags

FLAGS = flags.FLAGS

MAX = 4


class Log(logging.Logger):
	def setLevel(self, level):
		"""
		setLevel: sets the verbosity level of the logger
		@param: level	verbosity level setting
		"""
		try:
			super().setLevel(MAX - int(level) + 1)
		except:
			super().setLevel(level.upper())
		self.log(DEBUG, 'Log level set to %s', level)

	def getLevel(self):
		return MAX - int(self.level) + 1

	def VLOG(self, level, msg, *args, **kwargs):
		"""
		logs a log message at a given level with a given message
		@param: level	log level for message
		@param: msg		message that gets logged
		"""
		level = MAX - level + 1
		if not isinstance(level, int):
			if logging.raiseExceptions:
				raise TypeError("level must be an integer")
			else:
				return
		if self.isEnabledFor(level):
			self._log(level, msg, args, **kwargs)


def getLogger(name="root", log_level=False, log_type=False):
	"""
    Return a logger with the specified name, creating it if necessary.

    If no name is specified, return the root logger.
	@param: name		the name of the logger
	@param:	loglevel	log level for message
	@param: logType		formatting type
	@returns: logger	the logger
    """
	if name not in Log.manager.loggerDict:
		Log.manager.loggerDict[name] = Log(name)

	logger = Log.manager.getLogger(name)

	if log_level or log_type:
		handler = logging.StreamHandler()
		# format = logging.Formatter('%(levelname)s d:%(asctime)s %(name)-18s:%(lineno)-4s | %(message)s', '%Y.%m.%d
		# %H:%M:%S')
		logger.removeHandler(handler)

		logger.setLevel(log_level)

		handler.setFormatter(LoggerFormatter(log_type))

		logger.addHandler(handler)

	return logger


def format_message(record):
	"""
	formats the log message using the log record

	@param: record				log record
	@returns: record_message	the final log message
	"""
	try:
		record_message = '%s' % (record.msg % record.args)
	except TypeError:
		record_message = record.msg
	return record_message


class LoggerFormatter(logging.Formatter):
	LEVEL_MAP = {
		logging.FATAL: 'F',  # FATAL is alias of CRITICAL
		logging.ERROR: 'E',
		logging.WARN: 'W',
		logging.INFO: 'I',
		logging.DEBUG: 'D'
	}

	def __init__(self, log_type="cpp"):
		logging.Formatter.__init__(self)
		self.log_type = log_type

	def format(self, record):
		"""
		formats the input message with simplified C++ glog standards
		
		@param: record	contains the message logging information
		@returns: log record containing record and formatted message
		"""
		try:
			level = LoggerFormatter.LEVEL_MAP[record.levelno]
		except KeyError:
			level = str(MAX - record.levelno + 1)
		date = time.localtime(record.created)
		date_usec = (record.created - int(record.created)) * 1e3
		delim = " | " if self.log_type == "pretty" else "] "
		if self.log_type == "pretty":
			record_message = '%c d:%02d.%02d %02d:%02d:%02d.%03d %16s:%4d%s%s' % (
				level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
				date.tm_sec, date_usec,
				record.filename,
				record.lineno,
				delim,
				format_message(record))
		else:
			record_message = '%c%02d%02d %02d:%02d:%02d.%06d %s %s:%d%s%s' % (
				level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
				date.tm_sec, date_usec,
				record.process if record.process is not None else '?????',
				record.filename,
				record.lineno,
				delim,
				format_message(record))

		message = record_message
		try:
			start = message.index("%%%") + 3
			length = message.index(delim) + len(delim)
			newMessage = " " * length + message[start:].lstrip()
		except:
			message = message.split("\n")
			length = message[0].index(delim) + len(delim)
			newMessage = message[0]
			message = message[1:]
			for string in message:
				newMessage += "\n" + " " * length + string
		record.getMessage = lambda: newMessage
		return logging.Formatter.format(self, record)


logger = logging.getLogger()
handler = logging.StreamHandler()


def setLevel(level):
	try:
		logger.setLevel(MAX - int(level) + 1)
	except:
		logger.setLevel(level.upper())
	logger.debug('Log level set to %s', level)


debug = logging.debug
info = logging.info
warning = logging.warning
warn = logging.warning
error = logging.error
exception = logging.exception
fatal = logging.fatal
log = logger.log

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
WARN = logging.WARN
ERROR = logging.ERROR
FATAL = logging.FATAL

_level_names = {
	DEBUG: 'DEBUG',
	INFO: 'INFO',
	WARN: 'WARN',
	ERROR: 'ERROR',
	FATAL: 'FATAL'
}

_level_letters = [name[0] for name in _level_names.values()]

GLOG_PREFIX_REGEX = (
	                    r"""
    (?x) ^
    (?P<severity>[%s])
    (?P<month>\d\d)(?P<day>\d\d)\s
    (?P<hour>\d\d):(?P<minute>\d\d):(?P<second>\d\d)
    \.(?P<microsecond>\d{6})\s+
    (?P<process_id>-?\d+)\s
    (?P<filename>[a-zA-Z<_][\w._<>-]+):(?P<line>\d+)
    \]\s
    """) % ''.join(_level_letters)
"""Regex you can use to parse glog line prefixes."""

handler.setFormatter(LoggerFormatter())
logger.addHandler(handler)


class CaptureWarningsFlag(flags.BooleanFlag):
	def __init__(self):
		flags.BooleanFlag.__init__(self, 'glog_capture_warnings', True,
		                           "Redirect warnings to log.warn messages")

	def Parse(self, arg):
		flags.BooleanFlag.Parse(self, arg)
		logging.captureWarnings(self.value)


flags.DEFINE_flag(CaptureWarningsFlag())


class VerbosityParser(flags.ArgumentParser):
	"""Sneakily use gflags parsing to get a simple callback."""

	def Parse(self, arg):
		try:
			intarg = int(arg)
			# Look up the name for this level (DEBUG, INFO, etc) if it exists
			try:
				level = logging._levelNames.get(intarg, intarg)
			except AttributeError:  # This was renamed somewhere b/w 2.7 and 3.4
				level = logging._levelToName.get(intarg, intarg)
		except ValueError:
			level = arg
		setLevel(level)
		return level


flags.DEFINE(
	parser=VerbosityParser(),
	serializer=flags.ArgumentSerializer(),
	name='verbosity',
	default=logging.INFO,
	help='Logging verbosity')


# Define functions emulating C++ glog check-macros
# https://htmlpreview.github.io/?https://github.com/google/glog/master/doc/glog.html#check

def format_stacktrace(stack):
	"""Print a stack trace that is easier to read.

    * Reduce paths to basename component
    * Truncates the part of the stack after the check failure
    """
	lines = []
	for _, f in enumerate(stack):
		fname = os.path.basename(f[0])
		line = "\t%s:%d\t%s" % (fname + "::" + f[2], f[1], f[3])
		lines.append(line)
	return lines


class FailedCheckException(AssertionError):
	"""Exception with message indicating check-failure location and values."""


def check_failed(message):
	stack = traceback.extract_stack()
	stack = stack[0:-2]
	stacktrace_lines = format_stacktrace(stack)
	filename, line_num, _, _ = stack[-1]

	try:
		raise FailedCheckException(message)
	except FailedCheckException:
		log_record = logger.makeRecord('CRITICAL', 50, filename, line_num,
		                               message, None, None)
		handler.handle(log_record)

		log_record = logger.makeRecord('DEBUG', 10, filename, line_num,
		                               'Check failed here:', None, None)
		handler.handle(log_record)
		for line in stacktrace_lines:
			log_record = logger.makeRecord('DEBUG', 10, filename, line_num,
			                               line, None, None)
			handler.handle(log_record)
		raise
	return


def check(condition, message=None):
	"""Raise exception with message if condition is False."""
	if not condition:
		if message is None:
			message = "Check failed."
		check_failed(message)


def check_eq(obj1, obj2, message=None):
	"""Raise exception with message if obj1 != obj2."""
	if obj1 != obj2:
		if message is None:
			message = "Check failed: %s != %s" % (str(obj1), str(obj2))
		check_failed(message)


def check_ne(obj1, obj2, message=None):
	"""Raise exception with message if obj1 == obj2."""
	if obj1 == obj2:
		if message is None:
			message = "Check failed: %s == %s" % (str(obj1), str(obj2))
		check_failed(message)


def check_le(obj1, obj2, message=None):
	"""Raise exception with message if not (obj1 <= obj2)."""
	if obj1 > obj2:
		if message is None:
			message = "Check failed: %s > %s" % (str(obj1), str(obj2))
		check_failed(message)


def check_ge(obj1, obj2, message=None):
	"""Raise exception with message unless (obj1 >= obj2)."""
	if obj1 < obj2:
		if message is None:
			message = "Check failed: %s < %s" % (str(obj1), str(obj2))
		check_failed(message)


def check_lt(obj1, obj2, message=None):
	"""Raise exception with message unless (obj1 < obj2)."""
	if obj1 >= obj2:
		if message is None:
			message = "Check failed: %s >= %s" % (str(obj1), str(obj2))
		check_failed(message)


def check_gt(obj1, obj2, message=None):
	"""Raise exception with message unless (obj1 > obj2)."""
	if obj1 <= obj2:
		if message is None:
			message = "Check failed: %s <= %s" % (str(obj1), str(obj2))
		check_failed(message)


def check_notnone(obj, message=None):
	"""Raise exception with message if obj is None."""
	if obj is None:
		if message is None:
			message = "Check failed: Object is None."
		check_failed(message)
