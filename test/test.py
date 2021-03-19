import glog as log
import argparse
# FLAGS = flags.FLAGS

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

# flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

#   log.log(4, "four, least descriptive")
#   log.log(3, "three")
#   log.log(2, "two")
#   log.log(1, "one, most descriptive")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', help='verbosity level for file prints', type=int)
    args = parser.parse_args()

    print('The value of myflag is %s' % args.verbosity)
    log.setLevel(args.verbosity)
    log.VLOG(4, "four, most descriptive")
    log.VLOG(3, "three")
    log.VLOG(2, "two")
    log.VLOG(1, "one, least descriptive")