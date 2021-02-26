import glog as log
from absl import app
from absl import flags
# FLAGS = flags.FLAGS
MAX = 4

def VLOG(vLevel,string):
    log.log(MAX-vLevel+1,string)
def setLevel(level):
    log.logger.setLevel(MAX-level+1)
    log.logger.debug('Log level set to %s', level)

log.info("It works.")
log.warn("Something not ideal")
log.error("Something went wrong")
log.fatal("AAAAAAAAAAAAAAA!")

# def main(argv):
#   if FLAGS.debug:
#     print('non-flag arguments:', argv)
#   print('Happy Birthday', FLAGS.name)
#   if FLAGS.age is not None:
#     print('You are %d years old, and your job is %s' % (FLAGS.age, FLAGS.job))

FLAGS = flags.FLAGS

# flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

def main(argv):
    # if FLAGS.debug:
    #     print('non-flag arguments:', argv)
    print('The value of myflag is %s' % FLAGS.verbosity)
    setLevel(FLAGS.verbosity)
    VLOG(4, "four, most descriptive")
    VLOG(3, "three")
    VLOG(2, "two")
    VLOG(1, "one, least desciptive")
#   log.log(4, "four, least descriptive")
#   log.log(3, "three")
#   log.log(2, "two")
#   log.log(1, "one, most desciptive")


if __name__ == '__main__':
  app.run(main)