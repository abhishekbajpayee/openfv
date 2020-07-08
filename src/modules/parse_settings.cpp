//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2017 Abhishek Bajpayee
//
// This file is part of OpenFV.
//
// OpenFV is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License version 2 as published by the Free Software Foundation.
//
// OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License version 2 for more details.
//
// You should have received a copy of the GNU General Public License version 2 along with
// OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

#include "std_include.h"

#include "refocusing.h"
#include "tools.h"

using namespace cv;
using namespace std;

boost::program_options::options_description get_options() {
    
    namespace po = boost::program_options;
    po::options_description desc("Allowed config file options");

    desc.add_options()
        ("use_gpu", po::value<int>()->default_value(0), "ON to use GPU")
        ("mult", po::value<int>()->default_value(0), "ON to use multiplicative method")
        ("mult_exp", po::value<double>()->default_value(1.0), "Multiplicative method exponent")
        ("minlos", po::value<int>()->default_value(0), "ON to use minimum line of sight method")
        ("nlca", po::value<int>()->default_value(0), "ON to use nonlinear contrast adjustment")
        ("nlca_fast", po::value<int>()->default_value(0), "ON to use fast nonlinear contrast adjustment")
        ("nlca_win", po::value<int>()->default_value(32), "NLCA window size")
        ("delta", po::value<double>()->default_value(0.1), "NLCA delta")
        ("weighting", po::value<int>()->default_value(0), "ON to use weightin (0 -> -1)")
        ("hf_method", po::value<int>()->default_value(0), "ON to use HF method")
        ("mtiff", po::value<int>()->default_value(0), "ON if data is in multipage tiff files")
        ("frames", po::value<string>()->default_value(""), "Array of values in format start, end, skip")
        ("calib_file_path", po::value<string>()->default_value(""), "calibration file to use")
        ("images_path", po::value<string>()->default_value(""), "path where data is located")
        ("shifts", po::value<string>()->default_value(""), "path where data is located")
        ("resize_images", po::value<int>()->default_value(0), "ON to resize all input images")
        ("rf", po::value<double>()->default_value(1.0), "Factor to resize input images by")
        ("undistort", po::value<int>()->default_value(0), "ON to undistort images")

        ("save_path", po::value<string>()->default_value(""), "path where data is saved")
        ("zmin", po::value<double>()->default_value(0), "zmin")
        ("zmax", po::value<double>()->default_value(10), "zmax")
        ("manual_dz", po::value<int>()->default_value(0), "flag to enable manual specification of dz")
        ("dz", po::value<double>()->default_value(0.1), "dz (manual)")
        ("thresh", po::value<double>()->default_value(2.5), "threshold (std devs above mean)")

        ;

    return desc;

}

void parse_reconstruction_settings(string filename, reconstruction_settings &settings, bool h) {

    namespace po = boost::program_options;

    po::options_description desc = get_options();
 
    if (h) {
        cout<<desc;
        exit(1);
    }

    po::variables_map vm;
    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);
 
    settings.zmin = vm["zmin"].as<double>();
    settings.zmax = vm["zmax"].as<double>();
    settings.manual_dz = vm["manual_dz"].as<int>();
    if (settings.manual_dz)
        settings.dz = vm["dz"].as<double>();
    settings.thresh = vm["thresh"].as<double>();

    boost::filesystem::path saveP(vm["save_path"].as<string>());
    if(saveP.string().empty()) {
        LOG(FATAL)<<"save_path is a REQUIRED variable";
    }
    if (saveP.is_absolute()) {
        settings.save_path = saveP.string();
    } else {
        boost::filesystem::path file_path = boost::filesystem::canonical(filename, boost::filesystem::current_path());
        file_path.remove_leaf() /= saveP.string();
        settings.save_path = file_path.string();
    }
    if (*settings.save_path.rbegin() != '/')
        settings.save_path += '/';

}

void parse_refocus_settings(string filename, refocus_settings &settings, bool h) {

    namespace po = boost::program_options;

    po::options_description desc = get_options();
 
    if (h) {
        cout<<desc;
        exit(1);
    }

    po::variables_map vm;
    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);

    settings.use_gpu = vm["use_gpu"].as<int>();
    settings.hf_method = vm["hf_method"].as<int>();
    settings.mtiff = vm["mtiff"].as<int>();
    settings.mult = vm["mult"].as<int>();
    settings.mult_exp = vm["mult_exp"].as<double>();
    settings.minlos = vm["minlos"].as<int>();
    settings.nlca = vm["nlca"].as<int>();
    settings.nlca_fast = vm["nlca_fast"].as<int>();
    settings.nlca_win = vm["nlca_win"].as<int>();
    settings.delta = vm["delta"].as<double>();
    settings.weighting = vm["weighting"].as<int>();
    settings.resize_images = vm["resize_images"].as<int>();
    settings.rf = vm["rf"].as<double>();
    settings.undistort = vm["undistort"].as<int>();

    vector<int> frames;
    stringstream frames_stream(vm["frames"].as<string>());
    int i;
    while (frames_stream >> i) {
        frames.push_back(i);

        if(frames_stream.peek() == ',' || frames_stream.peek() == ' ') {
            frames_stream.ignore();
        }
    }
    if (frames.size() == 0) {
        settings.all_frames = 1;
        settings.skip = 0;
    } else if (frames.size() == 1) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(0);
        settings.skip = 0;
        settings.all_frames = 0;
    } else if (frames.size() == 2) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(1);
        settings.skip = 0;
        settings.all_frames = 0;
    } else if (frames.size() >= 3) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(1);
        settings.skip = frames.at(2);
        settings.all_frames = 0;
    }
    if (settings.start_frame<0) {
        LOG(FATAL)<<"Can't have starting frame less than 0. Terminating..."<<endl;
    }

    // Reading time shift values
    vector<int> shifts;
    stringstream shifts_stream(vm["shifts"].as<string>());
    while (shifts_stream >> i) {
        shifts.push_back(i);
        if(shifts_stream.peek() == ',' || shifts_stream.peek() == ' ') {
            shifts_stream.ignore();
        }
    }
    settings.shifts = shifts;


    // settings.all_frames = vm["all_frames"].as<int>();
    // if (!settings.all_frames) {
    //     settings.start_frame = vm["start_frame"].as<int>();
    //     settings.end_frame = vm["end_frame"].as<int>();
    // }
    // settings.upload_frame = vm["upload_frame"].as<int>();

    boost::filesystem::path calibP(vm["calib_file_path"].as<string>());
    if(calibP.string().empty()) {
        LOG(FATAL)<<"calib_file_path is a REQUIRED variable";
    }
    if (calibP.is_absolute()) {
        settings.calib_file_path = calibP.string();
    } else {
        boost::filesystem::path config_file_path = boost::filesystem::canonical(filename, boost::filesystem::current_path());
        config_file_path.remove_leaf() /= calibP.string();
        settings.calib_file_path = config_file_path.string();
    }
    if (!boost::filesystem::exists(settings.calib_file_path))
        LOG(FATAL) << settings.calib_file_path << " does not seem to exist!";
    if (!boost::filesystem::is_regular_file(settings.calib_file_path))
        LOG(FATAL) << settings.calib_file_path << " does not seem to be a file! Is it a directory?";

    boost::filesystem::path imgsP(vm["images_path"].as<string>());
    if(imgsP.string().empty()) {
        LOG(FATAL)<<"images_path is a REQUIRED variable";
    }
    if (imgsP.is_absolute()) {
        settings.images_path = imgsP.string();
    } else {
        boost::filesystem::path config_file_path = boost::filesystem::canonical(filename, boost::filesystem::current_path());;
        config_file_path.remove_leaf() /= imgsP.string();
        settings.images_path = config_file_path.string();
    }
    if (!boost::filesystem::exists(settings.images_path))
        LOG(FATAL) << settings.images_path << " does not seem to exist!";
    if (!boost::filesystem::is_directory(settings.images_path))
        LOG(FATAL) << settings.images_path << " does not seem to be a directory! Is it a file?";
    if (*settings.images_path.rbegin() != '/')
        settings.images_path += '/';

}

/*
void parse_calibration_settings(string filename, calibration_settings &settings, bool h) {

    namespace po = boost::program_options;

    po::options_description desc("Allowed config file options");
    desc.add_options()
        ("images_path", po::value<string>()->default_value(""), "path where data is located")
        ("corners_file", po::value<string>()->default_value(""), "file where to write corners")
        ("refractive", po::value<int>()->default_value(0), "ON if calibration data is refractive")
        ("mtiff", po::value<int>()->default_value(0), "ON if data is in multipage tiff files")
        ("mp4", po::value<int>()->default_value(0), "ON if data is in mp4 files")
        ("distortion", po::value<int>()->default_value(0), "ON if radial distortion should be accounted for")
        ("hgrid", po::value<int>()->default_value(5), "Horizontal number of corners in the grid")
        ("vgrid", po::value<int>()->default_value(5), "Vertical number of corners in the grid")
        ("grid_size_phys", po::value<double>()->default_value(5), "Physical size of grid in [mm]")
        ("skip", po::value<int>()->default_value(1), "Number of frames to skip (used mostly when mtiff on)")
        ("frames", po::value<string>()->default_value(""), "Array of values in format start, end, skip")
        ("shifts", po::value<string>()->default_value(""), "Array of time shift values separated by commas")
        ("resize_images", po::value<int>()->default_value(0), "ON if calibration images should be resized")
        ("rf", po::value<double>()->default_value(1), "Factor by which to resize calibration images")
        ;

    if (h) {
        cout<<desc;
        exit(1);
    }

    po::variables_map vm;
    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);

    settings.grid_size = Size(vm["hgrid"].as<int>(), vm["vgrid"].as<int>());
    settings.grid_size_phys = vm["grid_size_phys"].as<double>();
    settings.refractive = vm["refractive"].as<int>();
    settings.distortion = vm["distortion"].as<int>();
    settings.mtiff = vm["mtiff"].as<int>();
    settings.skip = vm["skip"].as<int>();
    settings.resize_images = vm["resize_images"].as<int>();
    settings.rf = vm["rf"].as<double>();
    settings.mp4 = vm["mp4"].as<int>();

    vector<int> frames;
    stringstream frames_stream(vm["frames"].as<string>());
    int i;
    while (frames_stream >> i) {
        frames.push_back(i);

        if(frames_stream.peek() == ',' || frames_stream.peek() == ' ') {
            frames_stream.ignore();
        }
    }
    if (frames.size() == 3) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(1);
        settings.skip = frames.at(2);
    } else {
        LOG(FATAL)<<"frames expects 3 comma or space separated values";
    }

    if (settings.start_frame<0) {
        LOG(FATAL)<<"Can't have starting frame less than 0. Terminating..."<<endl;
    }

    // Reading time shift values
    vector<int> shifts;
    stringstream shifts_stream(vm["shifts"].as<string>());
    while (shifts_stream >> i) {
        shifts.push_back(i);
        if(shifts_stream.peek() == ',' || shifts_stream.peek() == ' ') {
            shifts_stream.ignore();
        }
    }
    settings.shifts = shifts;

    boost::filesystem::path imgsP(vm["images_path"].as<string>());
    if(imgsP.string().empty()) {
        LOG(FATAL)<<"images_path is a REQUIRED variable";
    }
    if (imgsP.is_absolute()) {
        settings.images_path = imgsP.string();
    } else {
        boost::filesystem::path config_file_path(filename);
        config_file_path.remove_leaf() /= imgsP.string();
        settings.images_path = config_file_path.string();
    }

    settings.corners_file_path = vm["corners_file"].as<string>();

}
*/
