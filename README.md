# wuwiak

RISC-V compiler of the objectively best programming language. Developed on [WWW21](https://warsztatywww.pl/2025/workshop/kompi/).

## Usage

Build compiler and run with
```sh
uv run wuwiakc -h
```

To compile a Wuwiak program run
```
uv run wuwiakc build File.src -o File.s
```

where `File.src` is the input file, and `File.s` is the output file.

### Running using rars

Make sure RARS is installed in the working dir:

```
git clone https://github.com/TheThirdOne/rars
```

Then, you can compile and execute a source file with

```
uv run wuwiakc run File.src
```
