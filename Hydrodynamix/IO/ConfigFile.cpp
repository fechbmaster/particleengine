/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Hydrodynamix
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#include "StdAfx.hpp"
#include "ConfigFile.hpp"

#include <fstream>

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace IO {
    ConfigFile::ConfigFile(const QString& filename, FileFormat format) {
        open(filename, format);
    }

    void ConfigFile::open(const QString& filename, FileFormat format) {
        setFileName(filename);
        setFileFormat(format);
        reload();
    }

    void ConfigFile::reload() {
        ptree.clear();

        switch (fileFormat) {
        case FileFormat::Info:
            boost::property_tree::read_info(fileName.toStdString(), ptree);
            break;
        case FileFormat::Ini:
            boost::property_tree::read_ini(fileName.toStdString(), ptree);
            break;
        case FileFormat::Xml:
            boost::property_tree::read_xml(fileName.toStdString(), ptree);
            break;
        case FileFormat::Json:
            boost::property_tree::read_json(fileName.toStdString(), ptree);
            break;
        }
    }

    void ConfigFile::flush() {
        std::fstream fileStrm(fileName.toStdString(), std::ios::out | std::ios::trunc);

        switch (fileFormat) {
        case FileFormat::Info:
            boost::property_tree::write_info(fileStrm, ptree);
            break;
        case FileFormat::Ini:
            boost::property_tree::write_ini(fileStrm, ptree);
            break;
        case FileFormat::Xml:
            boost::property_tree::write_xml(fileStrm, ptree);
            break;
        case FileFormat::Json:
            boost::property_tree::write_json(fileStrm, ptree);
            break;
        }

        fileStrm.flush();
        fileStrm.close();
    }

    void ConfigFile::setFileFormat(FileFormat format) {
        fileFormat = format;
        if (fileFormat == FileFormat::Unknown) {
            fileFormat = detectFileFormat();
        }
    }

    ConfigFile::FileFormat ConfigFile::detectFileFormat() const {
        if (fileName.endsWith(".info", Qt::CaseInsensitive))
            return FileFormat::Info;
        if (fileName.endsWith(".ini", Qt::CaseInsensitive))
            return FileFormat::Ini;
        if (fileName.endsWith(".xml", Qt::CaseInsensitive))
            return FileFormat::Xml;
        if (fileName.endsWith(".json", Qt::CaseInsensitive))
            return FileFormat::Json;
        return FileFormat::Info;
    }
}
