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
#include "../../IO/ConfigFile.hpp"

#include <QTest>

class ConfigFileTest : public QObject {
    Q_OBJECT

public:
    ConfigFileTest() = default;
    ~ConfigFileTest() = default;

private slots:
    void initTestCase();
    void testFileFormat();
    void testRead();
    void testReadFloat();
    void testReadDefault();
    void testWrite();
    void testConvert();

private:
    IO::ConfigFile configFile;
};

void ConfigFileTest::initTestCase() {
    configFile.open("test.ini");
}

void ConfigFileTest::testFileFormat() {
    QCOMPARE(configFile.getFileFormat(), IO::ConfigFile::Ini);
}

void ConfigFileTest::testRead() {
    auto value = configFile.read<QString>("tests.read");
    QCOMPARE(value, QString("success"));
}

void ConfigFileTest::testReadFloat() {
    auto value = configFile.read<float>("tests.readFloat");
    QVERIFY(abs(value - 1.23f) < 0.0001f);
}

void ConfigFileTest::testReadDefault() {
    auto value = configFile.read<QString>("tests.readDefault", "default");
    QCOMPARE(value, QString("default"));
}

void ConfigFileTest::testWrite() {
    configFile.write("tests.write", "success");
    configFile.reload();

    auto value = configFile.read<QString>("tests.write");
    QCOMPARE(value, QString("success"));
}

void ConfigFileTest::testConvert() {
    configFile.setFileName("test.xml");
    configFile.setFileFormat(IO::ConfigFile::Xml);
    configFile.flush();
}

QTEST_MAIN(ConfigFileTest)
#include "configfiletest.moc"
