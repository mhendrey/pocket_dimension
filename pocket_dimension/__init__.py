"""
Pocket dimension provides a memory-efficient, dense, random projection of sparse vectors
and then applies this to Term Frequency (TF) and Term Frequency, Inverse Document
Frequency (TF-IDF) data.

Copyright (C) 2022 Matthew Hendrey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
__version__ = "0.1.0"

from pocket_dimension.pocket_dimension import (
    random_projection,
    TFVectorizer,
    TFIDFVectorizer,
    best_delta,
)
