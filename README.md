# reed-solomon-erasure

C++23 implementation of Reed-Solomon erasure coding.

This repository started as the Rust crate
[`reed-solomon-erasure`](https://github.com/darrenldl/reed-solomon-erasure)
(v6.0.0) and has been fully rewritten in C++23. The C++ implementation is
wire-compatible with the Rust crate: encoding the same data produces
byte-identical parity shards (verified across x86-64/AVX2 and
AArch64/NEON). The original Rust sources remain available in the git history.

Erasure coding means errors are not directly detected or corrected, but
missing data pieces (shards) can be reconstructed given enough redundancy.
Error detection (e.g. checksums) must be implemented separately.

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # run the test suite
cmake --install build           # install headers + libs + CMake package
```

Both a static (`libreed_solomon_erasure.a`) and a shared
(`libreed_solomon_erasure.so`, SOVERSION 6) library are produced. Consume via
CMake as `rse::static` / `rse::shared` (in-tree) or
`find_package(reed_solomon_erasure)` (installed).

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `RSE_BUILD_TESTS` | `ON` | build the doctest-based test suite |
| `RSE_BUILD_BENCH` | `OFF` | build the bandwidth benchmark (`rse_bandwidth`) |

## SIMD

The GF(2^8) bulk multiply (`mul_slice` / `mul_slice_xor`) uses the
nibble-split `pshufb`/`tbl` technique from the original `simd_c` kernels:

- x86-64: SSSE3, AVX2, and AVX-512BW kernels are always compiled (each in its
  own translation unit with the matching `-m` flags) and the best one is
  selected at **runtime** via CPU detection. No `-march` flag is required.
- AArch64: NEON kernel (baseline, always available).
- Anything else: portable scalar table lookups.

`rse::gf8::simd_backend()` reports the selected backend
(`"avx512"`, `"avx2"`, `"ssse3"`, `"neon"`, `"scalar"`).

GF(2^16) uses the generic element-wise path, as in the Rust crate.

## Usage

```cpp
#include <rse/rse.hpp>

auto r = rse::galois_8::ReedSolomon::create(3, 2).value();   // 3 data + 2 parity

std::vector<std::vector<std::uint8_t>> shards = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},   // data
    {0, 0, 0}, {0, 0, 0},              // parity (overwritten)
};
r.encode(shards).value();
assert(r.verify(shards).value());

// Reconstruction from "option shards":
std::vector<std::optional<std::vector<std::uint8_t>>> opt(shards.begin(), shards.end());
opt[0] = std::nullopt;
opt[4] = std::nullopt;
r.reconstruct(opt).value();

// ... or in place with present flags:
bool present[] = {false, true, true, true, true};
shards[0] = {9, 9, 9};                 // corrupted
r.reconstruct(shards, present).value();
```

## API mapping (Rust → C++)

| Rust | C++ |
| --- | --- |
| `ReedSolomon::new(d, p) -> Result<_, Error>` | `ReedSolomon<F>::create(d, p) -> std::expected<_, Error>` |
| `encode`, `encode_sep`, `encode_single`, `encode_single_sep` | same names; accept any range of contiguous shards |
| `verify`, `verify_with_buffer` | same names, return `std::expected<bool, Error>` |
| `reconstruct(&mut [Option<Vec<u8>>])` | `reconstruct(std::vector<std::optional<...>>&)` |
| `reconstruct(&mut [(&mut [u8], bool)])` | `reconstruct(shards, std::span<const bool> present)` |
| `ShardByShard` | `ShardByShard<F>` |
| panics (slice length mismatch, invert non-square, div by 0) | exceptions (`std::invalid_argument`, `std::domain_error`) |
| build.rs table generation | `consteval` table generation in `galois_8.hpp` |

## Tests

The test suite is a port of the Rust crate's tests (`src/tests/mod.rs`,
`src/tests/galois_16.rs`, plus module tests in `galois_8.rs`, `galois_16.rs`,
`matrix.rs`, `errors.rs`). The `quickcheck!` properties run as random-input
loops; SIMD kernels are additionally checked against the scalar reference for
all 256 multipliers across lengths covering each vector width and tail.

## License

MIT. See [LICENSE](LICENSE); the implementation derives from work by
Darren Ldl (Rust crate), Nicolas Trangez and Klaus Post (SIMD kernels), and
Backblaze (Java Reed-Solomon, the origin of the coding approach).
