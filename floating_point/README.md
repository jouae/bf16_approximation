# BF16
## Test Method
### Dependencies
Libraries:
* MPFR (GNU Multiple Precision Floating-Point Reliable Library)
* GMP (GNU Multiple Precision Arithmetic Library)

```bash
sudo apt install build-essential libmpfr-dev libgmp-dev
```
### Running Tests
```
make test
```

Produces three log files:

```
.range1.log -> [0.140625, pi/2] From 0x3E10 to 0x3FC9
.range2.log -> [pi/2, +largest] From 0x3FC9 to 0x7F7F
.range3.log -> [-smallest, -largest] From 0x8080 to 0xFF7F
```

where `+largest/-largest` is the largest positive/negative normal BF16, that is `+largest=0x7F7F` and `-largest=0xFF7F`.

Each line in log files contains:
```
<absolute_difference>,<input_bits>,<reference_bits(MPRF)>, <approx_bits(bf16_sin)>
```