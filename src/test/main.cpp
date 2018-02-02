//
// Created by guli on 02/02/18.
//

#include <iostream>
#include <gmpxx.h>

int main(int, const char *const *argv) {
    const char *q = argv[1];
    const char *d = argv[2];

    mpz_class qq(q);
    mpz_class dd(d);
    mpz_class res = 0;
    mpz_tdiv_r(res.get_mpz_t(), qq.get_mpz_t(), dd.get_mpz_t());
    gmp_printf("mpz_tdiv_r - %Zd\n", res.get_mpz_t());
    mpz_tdiv_q(res.get_mpz_t(), qq.get_mpz_t(), dd.get_mpz_t());
    gmp_printf("mpz_tdiv_q - %Zd\n", res.get_mpz_t());


}