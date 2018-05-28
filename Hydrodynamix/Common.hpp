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

#ifndef COMMON_HPP
#define COMMON_HPP

#include <type_traits>

#ifndef NOEXCEPT
#define NOEXCEPT throw()
#endif

#define DEF_GETTER(function, variable) \
    auto function() const NOEXCEPT -> \
        decltype(variable) const& { return variable; }

#define DEF_SETTER(function, variable) \
    void function(decltype(variable) const& var) \
        NOEXCEPT { variable = var; }

#define DEF_GETTER_VIRTUAL(function, variable) \
    virtual auto function() const NOEXCEPT -> \
        decltype(variable) { return variable; }

#define DEF_SETTER_VIRTUAL(function, variable) \
    virtual void function(decltype(variable) const& var) \
        NOEXCEPT { variable = var; }

#define DEF_GETTER_SETTER(function, variable) \
    DEF_GETTER(get##function, variable) \
    DEF_SETTER(set##function, variable)

#endif
