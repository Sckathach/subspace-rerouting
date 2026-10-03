# Typst <3

- Tutorial: <https://typst.app/docs/tutorial/>
- GitHub: <https://github.com/typst/typst>
- VSCode extension: <https://marketplace.visualstudio.com/items?itemName=myriad-dreamin.tinymist>

## Install

This paper requires Typst **v0.13.0**. You can install the correct version using cargo (or using the latest via: <https://github.com/typst-community/typst-install>, though the latest won't compile the paper):

```sh
cargo install typst-cli@0.13.0
```

## Use

Compile with:

```sh
typst compile paper/main.typ
```

Compile on save:

```sh
typst watch paper/main.typ
```
