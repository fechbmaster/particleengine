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

#ifndef CONFIG_FILE_HPP
#define CONFIG_FILE_HPP

#include <QString>
#include <boost/property_tree/ptree.hpp>

namespace IO {
    class ConfigFile {
    public:
        enum FileFormat {
            Unknown,
            Info,
            Ini,
            Xml,
            Json
        };

    public:
        ConfigFile() = default;
        ConfigFile(const QString& filename, FileFormat format = Unknown);
        ~ConfigFile() = default;

    public: // Generic read/write methods
        template <typename ValueType>
        void write(const QString& key, const ValueType& value) {
            ptree.put(key.toStdString(), value);
            flush();
        }

        template <typename ValueType>
        ValueType read(const QString& key, const ValueType& defaultValue) const {
            if (auto value = ptree.get_optional<ValueType>(key.toStdString())) {
                return *value;
            }
            return defaultValue;
        }

        template <typename ValueType>
        ValueType read(const QString& key) const {
            return ptree.get<ValueType>(key.toStdString());
        }

    public: // Special methods for QString handling
        template <>
        void write<QString>(const QString& key, const QString& value) {
            write(key, value.toStdString());
        }

        template <>
        QString read<QString>(const QString& key, const QString& defaultValue) const {
            return QString::fromStdString(read(key, defaultValue.toStdString()));
        }

        template <>
        QString read<QString>(const QString& key) const {
            return QString::fromStdString(read<std::string>(key));
        }

    public:
        void flush();
        void reload();

        void open(const QString& filename, FileFormat format = Unknown);

    public:
        FileFormat getFileFormat() const {
            return fileFormat;
        }

        void setFileFormat(FileFormat format);

        const QString& getFileName() const {
            return fileName;
        }

        void setFileName(const QString& filename) {
            fileName = filename;
        }

    private:
        FileFormat detectFileFormat() const;

    private:
        QString fileName;
        FileFormat fileFormat = Unknown;
        boost::property_tree::ptree ptree;
    };
}

#endif
