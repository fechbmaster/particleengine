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

#ifndef RANDOM_SET_HPP
#define RANDOM_SET_HPP

#include <chrono>
#include <functional>
#include <algorithm>
#include <random>

namespace Utils {
    template <typename T>
    class RandomSet {
    public:
        RandomSet(size_t count)
            : randomSet(count) {}

        ~RandomSet() = default;

    public:
        void create(const T& min, const T& max) {
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            auto random = std::bind(std::uniform_real_distribution<T>(min, max), std::mt19937(seed));
            std::generate(randomSet.begin(), randomSet.end(), random);
        }

    public:
        const T* data() const {
            return randomSet.data();
        }

        std::size_t size() const {
            return randomSet.size();
        }

        QVector<T> toVector() const {
            return randomSet;
        }

    private:
        QVector<T> randomSet;
    };
}

#endif
