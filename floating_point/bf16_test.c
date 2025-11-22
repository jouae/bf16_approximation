/* bf16_test.c
 *
 * MPFR-based validation harness for bf16_sin.
 * Uses mpfr_sin (200-bit) as oracle, then rounds MPFR -> float -> BF16
 * and compares to your bf16_sin result.
 *
 * Requirements:
 *  - libmpfr, libgmp available at link time
 * 
 * Header of logs:
 *  absolute_error, input_bits (hex), ref_bits, approx_bits
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpfr.h>

#include "bf16.h"

int main(void)
{
    mpfr_t x_mp, y_mp;
    mpfr_init2(x_mp, 200);   /* 200-bit precision */
    mpfr_init2(y_mp, 200);

    FILE *f1 = fopen(".range1.log", "w");
    FILE *f2 = fopen(".range2.log", "w");
    FILE *f3 = fopen(".range3.log", "w");

    if (!f1 || !f2 || !f3) {
        fprintf(stderr, "Failed to open log files\n");
        return 1;
    }

    bf16_t i;
    int record_k;

    double global_max_abs = 0.0;
    unsigned long global_diff_count = 0;

    /* ----------------------------------------------------------
     * Range 1: [0.140625, pi/2]
	 * From 0x3E10 (0.140625) to 0x3FC9 (1.5703125, rounded pi/2)
     * ----------------------------------------------------------*/
    double max_abs = 0.0;
    unsigned long diff_count = 0;

    for (i.bits = 0x3E10; i.bits < 0x3FC9; i.bits++) {
        float x = bf16_to_fp32(i);

        mpfr_set_flt(x_mp, x, MPFR_RNDN);
        mpfr_sin(y_mp, x_mp, MPFR_RNDN);
        float mpfr_sinf = mpfr_get_flt(y_mp, MPFR_RNDN);

        /* Round MPFR result to BF16 properly by converting MPFR->float then float->bf16 */
        bf16_t ref_bf = fp32_to_bf16(mpfr_sinf);
        float ref_rounded_f = bf16_to_fp32(ref_bf);

        bf16_t approx_bf = bf16_sin(i, &record_k);
        float approx_f = bf16_to_fp32(approx_bf);

        float abs_err = fabsf(approx_f - ref_rounded_f);

        if (abs_err > max_abs) max_abs = abs_err;
        if (abs_err > 0.0f) diff_count++;

        fprintf(f1, "%f, 0x%04x, 0x%04x, 0x%04x\n",
                abs_err,
                (unsigned) i.bits, (unsigned) ref_bf.bits, (unsigned) approx_bf.bits);
    }

    printf("Test range [0.140625, pi/2]:\nMax difference = %.9g, Total %lu numbers difference.\n\n",
           max_abs, diff_count);
    if (max_abs > global_max_abs) global_max_abs = max_abs;
    global_diff_count += diff_count;

    /* -------------------------------------------------------------
     * Range 2: [pi/2, +largest]
	 * From 0x3FC9 (rounded pi/2) to 0x7F7F(Largest positive normal)
	 * Do range reduction from [pi/2, +largeset] to [0, pi/2]
     * -------------------------------------------------------------*/
    max_abs = 0.0;
    diff_count = 0;

    for (i.bits = 0x3FC9; i.bits < 0x7F7F; i.bits++) {
        float x = bf16_to_fp32(i);

        mpfr_set_flt(x_mp, x, MPFR_RNDN);
        mpfr_sin(y_mp, x_mp, MPFR_RNDN);
        float mpfr_sinf = mpfr_get_flt(y_mp, MPFR_RNDN);

        bf16_t ref_bf = fp32_to_bf16(mpfr_sinf);
        float ref_rounded_f = bf16_to_fp32(ref_bf);

        bf16_t approx_bf = bf16_sin(i, &record_k);
        float approx_f = bf16_to_fp32(approx_bf);

        float abs_err = fabsf(approx_f - ref_rounded_f);

        if (abs_err > max_abs) max_abs = abs_err;
        if (abs_err > 0.0f) diff_count++;

        fprintf(f2, "%f, 0x%04x, 0x%04x, 0x%04x\n",
                abs_err,
                (unsigned) i.bits, (unsigned) ref_bf.bits, (unsigned) approx_bf.bits);
    }

    printf("Test range [pi/2, +largest]:\nMax difference = %.9g, Total %lu numbers difference.\n\n",
           max_abs, diff_count);
    if (max_abs > global_max_abs) global_max_abs = max_abs;
    global_diff_count += diff_count;

    /* --------------------------------------------------------------------------
     * Range 3: [-smallest, -largest]
	 * From 0x8080 (smallest negative normal) to 0xFF7F (largest negative normal)
	 * Do range reduction from [-smallest, -largest] to [0, pi/2]
     * --------------------------------------------------------------------------*/
    max_abs = 0.0;
    diff_count = 0;

    for (i.bits = 0x8080; i.bits < 0xFF7F; i.bits++) {
        float x = bf16_to_fp32(i);

        mpfr_set_flt(x_mp, x, MPFR_RNDN);
        mpfr_sin(y_mp, x_mp, MPFR_RNDN);
        float mpfr_sinf = mpfr_get_flt(y_mp, MPFR_RNDN);

        bf16_t ref_bf = fp32_to_bf16(mpfr_sinf);
        float ref_rounded_f = bf16_to_fp32(ref_bf);

        bf16_t approx_bf = bf16_sin(i, &record_k);
        float approx_f = bf16_to_fp32(approx_bf);

        float abs_err = fabsf(approx_f - ref_rounded_f);

        if (abs_err > max_abs) max_abs = abs_err;
        if (abs_err > 0.0f) diff_count++;

        fprintf(f3, "%f, 0x%04x, 0x%04x, 0x%04x\n",
                abs_err,
                (unsigned) i.bits, (unsigned) ref_bf.bits, (unsigned) approx_bf.bits);
    }

    printf("Test range [-smallest, -largest]:\nMax difference = %.9g, Total %lu numbers difference.\n",
           max_abs, diff_count);
    if (max_abs > global_max_abs) global_max_abs = max_abs;
    global_diff_count += diff_count;

    fclose(f1);
    fclose(f2);
    fclose(f3);

    mpfr_clear(x_mp);
    mpfr_clear(y_mp);

    return 0;
}
