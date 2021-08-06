import csv
import string
import math
import argparse

# Python script to convert from DLT point file to text file for openfv body tracking

def main(datapath, infile, outfile):

        #PATHS AND VARIABLES TO CONFIGURE
        # path = folder where DLT data is found with end /
        # example
        # path = '/home/leahm/localdata/processed71816/tracked/fish1jump10/'

        # infile = input filename from DLT, typically ends xypts.csv
        # infile = 'DLT_peduncle_xypts.csv'

        # outfile = output filename, must include .txt ending
        # example
        # outfile = 'fish1jump10peduncle.txt'

        # factor rescales points during text file generation (i.e., if the DLT images were resized before digitizing, 
        # NOT COMMONLY CHANGED
        # example: factor = 2 would rescale all coordinates by multiplying by 2)
        factor = 1

        #flipy denotes whether to switch from DLT coordinates (y+ up) to image coordinates (y+ down) when converting the file. 
        #flipy = 1 will flip coordinates, and requires the y-dimension of the source images (+1 to fix indexing) as an input. 
        #flipy = 0 will leave the data in DLT coordinates
        flipy = 1
        sy = 801

        # paths to input and output files, currently defaults to storing output .txt file in same directory
        inpath = datapath+infile
        outpath = datapath+outfile

        # create output file
        f = open(outpath, 'w')

        # read in .csv file from DLT input
        with open(inpath, 'r') as csvfile:

                # read in and get size of data
                rdr = csv.reader(csvfile, delimiter=',')
                #hdr = rdr.next()
                hdr = next(rdr)
                ncol = len(hdr)
                ref_hdr = hdr[ncol-1]

                # parsing of DLT default header rows
                pt_st = ref_hdr.index('pt')
                cam_st = ref_hdr.index('cam')
                end_st = ref_hdr.index('_Y')

                # more data sizing
                npts = ref_hdr[pt_st+2:cam_st-1]
                ncams = ref_hdr[cam_st+3:end_st]
                print(ncams+' cameras and ' +npts+' points')

                # list number of cameras and points in output text file
                f.write(ncams+'\n'+npts+'\n')

                # loop over and write all data
                for row in rdr:
                        tmp = row
                        #UNCOMMENT below for debugging
                        #print tmp
                        for col in range(0,ncol):
                                #UNCOMMENT below for debugging
                                #print tmp[col]
                                val = float(tmp[col])
                                flag = math.isnan(val)
                                if flag:
                                        #override NaNs
                                        val=0
                                else:
                                        #check if x or y column w/odd or even check on column index and check if y data needs to be flipped
                                        if col & 1 and flipy:
                                                val = sy+1-factor*val
                                        else:
                                                val = factor*val
                                        #UNCOMMENT below for debugging
                                        #print tmp[col]
		
                                f.write(str(val)+'\t')
                        f.write('\n')

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument('--datapath', default ="./", help='')
        parser.add_argument('--infile', required=True)
        parser.add_argument('--outfile', required=True)

        args = parser.parse_args()

        if args.datapath[-1] is not '/':
                args.datapath += '/'

main(args.datapath, args.infile, args.outfile)
