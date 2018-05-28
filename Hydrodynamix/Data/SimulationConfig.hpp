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

#ifndef SIMULATION_CONFIG_HPP
#define SIMULATION_CONFIG_HPP

#include "PropertyStore.hpp"

namespace Data {
    class SimulationConfig : public QObject {
        Q_OBJECT

    public:
        SimulationConfig(QObject* parent = nullptr)
            : QObject(parent) {}

        ~SimulationConfig() = default;

    public:
        void setGeneralProperties(const GeneralProperties& props);
        void setPhysicalProperties(const PhysicalProperties& props);
        void setComputeProperties(const ComputeProperties& props);

    public:
        const GeneralProperties& getGeneralProperties() const {
            return propertyStore.generalProperties;
        }

        const PhysicalProperties& getPhysicalProperties() const {
            return propertyStore.physicalProperties;
        }

        const ComputeProperties& getComputeProperties() const {
            return propertyStore.computeProperties;
        }

    signals:
        void generalPropertiesChanged(const PropertyStore&);
        void physicalPropertiesChanged(const PropertyStore&);
        void computePropertiesChanged(const PropertyStore&);

    private:
        PropertyStore propertyStore;

    public:
        DEF_GETTER(getPropertyStore, propertyStore);
    };
}

Q_DECLARE_METATYPE(Data::PropertyStore);

#endif
