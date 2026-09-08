# Third-party notices

## MicroPython SSD1306 driver lineage

`ssd1306.py` is derived from the MicroPython SSD1306 driver lineage. The
driver originated in the official `micropython/micropython` repository and
was moved to the official `micropython/micropython-lib` repository.

- Original driver commit:
  <https://github.com/micropython/micropython/commit/73bc0c24ab6419b3f41cb272286f99700846ab33>
- Last substantive main-repository source used for the move lineage:
  <https://github.com/micropython/micropython/commit/bc7822d8e95c40a9d5e403fd22c82b1bbad53b8b>
- Official move commit:
  <https://github.com/micropython/micropython-lib/commit/a5e2f3239c8023cf8d009fafa948d7f139cb87ff>
- Current pinned official reference:
  <https://github.com/micropython/micropython-lib/blob/a08087249fda8a7994f7c54ccaad29fb9fcc448a/micropython/drivers/display/ssd1306/ssd1306.py>

The file introduced by the official move is byte-identical to the cited last
substantive source in the main repository. The exact upstream revision copied
into this project and the identity of the local modifier are not established.

The `micropython-lib` root license is a composite notice covering modules from
multiple sources and under multiple licenses. The MIT conclusion for this
derivative is based on the original MicroPython MIT license and the official
byte-identical move lineage; it is not a claim that every `micropython-lib`
module or the whole collection has one license.

## Original MicroPython MIT notice

<!-- BEGIN EXACT MICROPYTHON MIT NOTICE -->
The MIT License (MIT)

Copyright (c) 2013, 2014 Damien P. George

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
<!-- END EXACT MICROPYTHON MIT NOTICE -->

## Current micropython-lib MIT notice

<!-- BEGIN EXACT MICROPYTHON-LIB MIT NOTICE -->
The MIT License (MIT)

Copyright (c) 2013, 2014 micropython-lib contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
<!-- END EXACT MICROPYTHON-LIB MIT NOTICE -->
