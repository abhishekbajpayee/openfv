#include "std_include.h"

#include "refocusing.h"
#include "typedefs.h"

using namespace cv;
using namespace std;

/*! Function to parse a refocus settings file and return a refocus_settings variable
  which can be directly passed to an saRefocus constructor.
  \param filename Full path to configuration file
  \param settings Destination refocus_settings variable
  \param help Flag to show all options as a help menu instead of parsing data
*/
void parse_refocus_settings(string filename, refocus_settings &settings, bool help);
