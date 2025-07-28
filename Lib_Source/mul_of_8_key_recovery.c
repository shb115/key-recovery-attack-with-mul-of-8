#include <stdint.h>
#include <memory.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include <time.h>
#include "mul_of_8_key_recovery.h"

#define AES128_128_ROUND 10
#define AES128_192_ROUND 12
#define AES128_256_ROUND 14

#define FAIL 0
#define SUCC 1
#define NO_PAIRS 2

int num_active_sboxes_in = 4;
int num_pasive_sboxes_ou = 8;

int in_active_indexes[4];
int ou_pasive_indexes[8];

typedef uint8_t ct_t[16];

typedef struct {
	uint8_t * pt;
	ct_t      ct;
}pair_t;

static __m128i aes_128_key_expansion(__m128i key, __m128i keygened) {
	keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3, 3, 3, 3));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
	return _mm_xor_si128(key, keygened);
}

#define AES_128_key_exp(k, rcon) aes_128_key_expansion(k, _mm_aeskeygenassist_si128(k, rcon))

int comp_states(const void * first, const void * second)
{
	pair_t * first_ptr = (pair_t *)first;
	pair_t * second_ptr = (pair_t *)second;
	int i;

	for (i = 0; i < num_pasive_sboxes_ou; i++)
	{
		uint8_t first_val = (*first_ptr).ct[ou_pasive_indexes[i]];
		uint8_t second_val = (*second_ptr).ct[ou_pasive_indexes[i]];

		if (first_val == second_val)
			continue;
		else if (first_val > second_val)
			return 1;
		else
			return -1;
	}
	return 0;
}

static const uint8_t sbox[256] = {
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

#define getSBoxValue(num)		(sbox[(num)])

#define xtime(x) ((x << 1) ^ (((x >> 7) & 1) * 0x1b))

#define Multiply(x, y)                                \
      (  ((y & 1) * x) ^                              \
      ((y>>1 & 1) * xtime(x)) ^                       \
      ((y>>2 & 1) * xtime(xtime(x))) ^                \
      ((y>>3 & 1) * xtime(xtime(xtime(x)))) ^         \
      ((y>>4 & 1) * xtime(xtime(xtime(xtime(x))))))   \

static void MixColumns(uint8_t* state)
{
	int i;
	uint8_t a, b, c, d;

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];

	state[0] = Multiply(a, 0x02) ^ Multiply(b, 0x03) ^ Multiply(c, 0x01) ^ Multiply(d, 0x01);
	state[1] = Multiply(a, 0x01) ^ Multiply(b, 0x02) ^ Multiply(c, 0x03) ^ Multiply(d, 0x01);
	state[2] = Multiply(a, 0x01) ^ Multiply(b, 0x01) ^ Multiply(c, 0x02) ^ Multiply(d, 0x03);
	state[3] = Multiply(a, 0x03) ^ Multiply(b, 0x01) ^ Multiply(c, 0x01) ^ Multiply(d, 0x02);
}

void MUL_OF_8_DISTINGUISHER_FOUND_PAIRS(uint8_t mk[16], uint8_t state[16], int32_t round)
{
	uint8_t round_idx = 0;
	__m128i RoundKey[AES128_128_ROUND + 1];
	__m128i tmp_state;
	pair_t* dat_tab = NULL;
	uint64_t dat_idx;
	uint64_t dat = 0, a, cnt;
	int idx;
	int i;	
	uint64_t num_dat = (1ULL) << (32);
	uint64_t found_pairs_num = 0;
	uint8_t* active_val_ptr = (uint8_t*)&dat_idx;

	// KeyExpansion()
	RoundKey[0]  = _mm_loadu_si128((const __m128i*) mk);
	RoundKey[1]  = AES_128_key_exp(RoundKey[0], 0x01);
	RoundKey[2]  = AES_128_key_exp(RoundKey[1], 0x02);
	RoundKey[3]  = AES_128_key_exp(RoundKey[2], 0x04);
	RoundKey[4]  = AES_128_key_exp(RoundKey[3], 0x08);
	RoundKey[5]  = AES_128_key_exp(RoundKey[4], 0x10);
	RoundKey[6]  = AES_128_key_exp(RoundKey[5], 0x20);
	RoundKey[7]  = AES_128_key_exp(RoundKey[6], 0x40);
	RoundKey[8]  = AES_128_key_exp(RoundKey[7], 0x80);
	RoundKey[9]  = AES_128_key_exp(RoundKey[8], 0x1B);
	RoundKey[10] = AES_128_key_exp(RoundKey[9], 0x36);

	dat_tab = (pair_t *)malloc(sizeof(pair_t)* num_dat);

	srand(time(NULL));

	in_active_indexes[0] = 0;
	in_active_indexes[1] = 5;
	in_active_indexes[2] = 10;
	in_active_indexes[3] = 15;

	for (dat_idx = 0; dat_idx < num_dat; dat_idx++)
	{
		//consider little-endian
		for (idx = 0; idx < num_active_sboxes_in; idx++) state[in_active_indexes[idx]] = active_val_ptr[idx];

		///////////////////////////////////////////////////////////////
		//Encryption
		tmp_state = _mm_loadu_si128((__m128i*) state);
		tmp_state = _mm_xor_si128(tmp_state, RoundKey[0]);
		for (round_idx = 1; round_idx < round; ++round_idx)	tmp_state = _mm_aesenc_si128(tmp_state, RoundKey[round_idx]);
		tmp_state = _mm_aesenclast_si128(tmp_state, RoundKey[round]);
		///////////////////////////////////////////////////////////////

		dat_tab[dat_idx].pt = (uint8_t*)malloc(sizeof(uint8_t) * num_active_sboxes_in);
		memcpy(dat_tab[dat_idx].pt, &dat, sizeof(uint8_t) * num_active_sboxes_in);
		_mm_storeu_si128((__m128i*) (dat_tab[dat_idx].ct), tmp_state);
	}

	printf("Data Generation Finished\n");
	
	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 1;
	ou_pasive_indexes[5] = 4;
	ou_pasive_indexes[6] = 11;
	ou_pasive_indexes[7] = 14;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 1 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;

			printf("P1 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].pt[i]);
			}
			printf("\n");
			printf("P2 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[dat_idx].pt[i]);
			}
			printf("\n");


			printf("C1 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].ct[i]);
			}
			printf("\n");
			printf("C2 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[dat_idx].ct[i]);
			}
			printf("\n");						
		}
		else cnt = 0;
	}
	
	printf("Totally Pairs are Found in Data 1\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 2 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;

			printf("P1 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].pt[i]);
			}
			printf("\n");
			printf("P2 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[dat_idx].pt[i]);
			}
			printf("\n");


			printf("C1 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].ct[i]);
			}
			printf("\n");
			printf("C2 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[dat_idx].ct[i]);
			}
			printf("\n");
		}
		else cnt = 0;
	}

	printf("Totally Pairs are Found in Data 2\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 3 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;

			printf("P1 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].pt[i]);
			}
			printf("\n");
			printf("P2 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[dat_idx].pt[i]);
			}
			printf("\n");


			printf("C1 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].ct[i]);
			}
			printf("\n");
			printf("C2 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[dat_idx].ct[i]);
			}
			printf("\n");
		}
		else cnt = 0;
	}

	printf("Totally Pairs are Found in Data 3\n");

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 4 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;

			printf("P1 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].pt[i]);
			}
			printf("\n");
			printf("P2 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[dat_idx].pt[i]);
			}
			printf("\n");


			printf("C1 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].ct[i]);
			}
			printf("\n");
			printf("C2 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[dat_idx].ct[i]);
			}
			printf("\n");
		}
		else cnt = 0;
	}

	printf("Totally %llu Pairs are Found in Data 4\n", found_pairs_num);

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 5 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;

			printf("P1 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].pt[i]);
			}
			printf("\n");
			printf("P2 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[dat_idx].pt[i]);
			}
			printf("\n");


			printf("C1 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].ct[i]);
			}
			printf("\n");
			printf("C2 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[dat_idx].ct[i]);
			}
			printf("\n");
		}
		else cnt = 0;
	}

	printf("Totally %llu Pairs are Found in Data 5\n", found_pairs_num);

	ou_pasive_indexes[0] = 2;
	ou_pasive_indexes[1] = 5;
	ou_pasive_indexes[2] = 8;
	ou_pasive_indexes[3] = 15;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 6 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;

			printf("P1 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].pt[i]);
			}
			printf("\n");
			printf("P2 : ");
			for (i = 0; i < num_active_sboxes_in; i++)
			{
				printf("%02X ", dat_tab[dat_idx].pt[i]);
			}
			printf("\n");


			printf("C1 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[(dat_idx - 1)].ct[i]);
			}
			printf("\n");
			printf("C2 : ");
			for (i = 0; i < 16; i++)
			{
				printf("%02X ", dat_tab[dat_idx].ct[i]);
			}
			printf("\n");
		}
		else cnt = 0;
	}

	printf("Totally %llu Pairs are Found in Data 6\n", found_pairs_num);

	free(dat_tab[num_dat - 1].pt);
	free(dat_tab);

	printf("##########################################################\n");
}

uint64_t MUL_OF_8_DISTINGUISHER_NUM_ONLY(uint8_t mk[16], uint8_t state[16], int32_t round)
{
	uint8_t round_idx = 0;
	__m128i RoundKey[AES128_128_ROUND + 1];
	__m128i tmp_state;
	pair_t* dat_tab = NULL;
	uint64_t dat_idx;
	uint64_t dat = 0, a, cnt;
	int idx;
	int i;
	uint64_t num_dat = (1ULL) << (32);
	uint64_t found_pairs_num = 0;
	uint8_t* active_val_ptr = (uint8_t*)&dat_idx;

	// KeyExpansion()
	RoundKey[0] = _mm_loadu_si128((const __m128i*) mk);
	RoundKey[1] = AES_128_key_exp(RoundKey[0], 0x01);
	RoundKey[2] = AES_128_key_exp(RoundKey[1], 0x02);
	RoundKey[3] = AES_128_key_exp(RoundKey[2], 0x04);
	RoundKey[4] = AES_128_key_exp(RoundKey[3], 0x08);
	RoundKey[5] = AES_128_key_exp(RoundKey[4], 0x10);
	RoundKey[6] = AES_128_key_exp(RoundKey[5], 0x20);
	RoundKey[7] = AES_128_key_exp(RoundKey[6], 0x40);
	RoundKey[8] = AES_128_key_exp(RoundKey[7], 0x80);
	RoundKey[9] = AES_128_key_exp(RoundKey[8], 0x1B);
	RoundKey[10] = AES_128_key_exp(RoundKey[9], 0x36);

	dat_tab = (pair_t*)malloc(sizeof(pair_t) * num_dat);

	srand(time(NULL));

	in_active_indexes[0] = 0;
	in_active_indexes[1] = 5;
	in_active_indexes[2] = 10;
	in_active_indexes[3] = 15;

	for (dat_idx = 0; dat_idx < num_dat; dat_idx++)
	{
		//consider little-endian
		for (idx = 0; idx < num_active_sboxes_in; idx++) state[in_active_indexes[idx]] = active_val_ptr[idx];

		///////////////////////////////////////////////////////////////
		//Encryption
		tmp_state = _mm_loadu_si128((__m128i*) state);
		tmp_state = _mm_xor_si128(tmp_state, RoundKey[0]);
		for (round_idx = 1; round_idx < round; ++round_idx)	tmp_state = _mm_aesenc_si128(tmp_state, RoundKey[round_idx]);
		tmp_state = _mm_aesenclast_si128(tmp_state, RoundKey[round]);
		///////////////////////////////////////////////////////////////

		dat_tab[dat_idx].pt = (uint8_t*)malloc(sizeof(uint8_t) * num_active_sboxes_in);
		memcpy(dat_tab[dat_idx].pt, &dat, sizeof(uint8_t) * num_active_sboxes_in);
		_mm_storeu_si128((__m128i*) (dat_tab[dat_idx].ct), tmp_state);
	}

	printf("Data Generation Finished\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 1;
	ou_pasive_indexes[5] = 4;
	ou_pasive_indexes[6] = 11;
	ou_pasive_indexes[7] = 14;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 1 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;
		}
		else cnt = 0;
	}

	printf("Totally Pairs are Found in Data 1\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 2 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;
		}
		else cnt = 0;
	}

	printf("Totally Pairs are Found in Data 2\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 3 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;
		}
		else cnt = 0;
	}

	printf("Totally Pairs are Found in Data 3\n");

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 4 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;
		}
		else cnt = 0;
	}

	printf("Totally %llu Pairs are Found in Data 4\n", found_pairs_num);

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 5 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;
		}
		else cnt = 0;
	}

	printf("Totally %llu Pairs are Found in Data 5\n", found_pairs_num);

	ou_pasive_indexes[0] = 2;
	ou_pasive_indexes[1] = 5;
	ou_pasive_indexes[2] = 8;
	ou_pasive_indexes[3] = 15;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	printf("Sorting Data 6 Finished\n");

	cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			cnt += 1;
			found_pairs_num += cnt;
		}
		else cnt = 0;
	}

	printf("Totally %llu Pairs are Found in Data 6\n", found_pairs_num);

	free(dat_tab[num_dat - 1].pt);
	free(dat_tab);

	printf("##########################################################\n");
}

uint64_t MUL_OF_8_KEY_RECOVERY_FIRST_DIAGONAL(uint8_t mk[16], uint8_t state[16], int32_t round)
{
	uint8_t round_idx = 0;
	__m128i RoundKey[AES128_128_ROUND + 1];
	__m128i tmp_state;
	pair_t* dat_tab = NULL;
	uint64_t dat_idx;
	uint64_t dat = 0, a, cnt, pair_cnt, cnt_idx;
	uint64_t pair_idx[2];
	int idx;
	int i;
	uint64_t num_dat = (1ULL) << (31);
	uint64_t found_pairs_num = 0;
	uint8_t* active_val_ptr = (uint8_t*)&dat_idx;
	uint64_t key_can;
	uint8_t temp1, temp2, temp3, temp4;
	int pair_flag = 0;

	// KeyExpansion()
	RoundKey[0] = _mm_loadu_si128((const __m128i*) mk);
	RoundKey[1] = AES_128_key_exp(RoundKey[0], 0x01);
	RoundKey[2] = AES_128_key_exp(RoundKey[1], 0x02);
	RoundKey[3] = AES_128_key_exp(RoundKey[2], 0x04);
	RoundKey[4] = AES_128_key_exp(RoundKey[3], 0x08);
	RoundKey[5] = AES_128_key_exp(RoundKey[4], 0x10);
	RoundKey[6] = AES_128_key_exp(RoundKey[5], 0x20);
	RoundKey[7] = AES_128_key_exp(RoundKey[6], 0x40);
	RoundKey[8] = AES_128_key_exp(RoundKey[7], 0x80);
	RoundKey[9] = AES_128_key_exp(RoundKey[8], 0x1B);
	RoundKey[10] = AES_128_key_exp(RoundKey[9], 0x36);

	dat_tab = (pair_t*)malloc(sizeof(pair_t) * num_dat);

	srand(time(NULL));

	in_active_indexes[0] = 0;
	in_active_indexes[1] = 5;
	in_active_indexes[2] = 10;
	in_active_indexes[3] = 15;

	for (dat_idx = 0; dat_idx < num_dat; dat_idx++)
	{
		//consider little-endian
		for (idx = 0; idx < num_active_sboxes_in; idx++) state[in_active_indexes[idx]] = active_val_ptr[idx];

		///////////////////////////////////////////////////////////////
		//Encryption
		tmp_state = _mm_loadu_si128((__m128i*) state);
		tmp_state = _mm_xor_si128(tmp_state, RoundKey[0]);
		for (round_idx = 1; round_idx < round; ++round_idx)	tmp_state = _mm_aesenc_si128(tmp_state, RoundKey[round_idx]);
		tmp_state = _mm_aesenclast_si128(tmp_state, RoundKey[round]);
		///////////////////////////////////////////////////////////////

		dat_tab[dat_idx].pt = (uint8_t*)malloc(sizeof(uint8_t) * num_active_sboxes_in);
		memcpy(dat_tab[dat_idx].pt, &dat_idx, sizeof(uint8_t) * num_active_sboxes_in);
		_mm_storeu_si128((__m128i*) (dat_tab[dat_idx].ct), tmp_state);
	}

	//printf("Data Generation Finished\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 1;
	ou_pasive_indexes[5] = 4;
	ou_pasive_indexes[6] = 11;
	ou_pasive_indexes[7] = 14;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 1 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;
			
			pair_cnt += 1;

			if (pair_cnt == 2)
			{				
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 1\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 2 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{			
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 2\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 3 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 3\n");

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 4 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 4\n", found_pairs_num);

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 5 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 5\n", found_pairs_num);

	ou_pasive_indexes[0] = 2;
	ou_pasive_indexes[1] = 5;
	ou_pasive_indexes[2] = 8;
	ou_pasive_indexes[3] = 15;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 6 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 6\n", found_pairs_num);
		
	// 할당했던 모든 pt 메모리 해제
	for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
		free(dat_tab[dat_idx].pt);
	}

	free(dat_tab);

	//printf("##########################################################\n");

	if (pair_flag == 0) return NO_PAIRS;

	return SUCC;
}

uint64_t MUL_OF_8_KEY_RECOVERY_SECOND_DIAGONAL(uint8_t mk[16], uint8_t state[16], int32_t round)
{
	uint8_t round_idx = 0;
	__m128i RoundKey[AES128_128_ROUND + 1];
	__m128i tmp_state;
	pair_t* dat_tab = NULL;
	uint64_t dat_idx;
	uint64_t dat = 0, a, cnt, pair_cnt, cnt_idx;
	uint64_t pair_idx[2];
	int idx;
	int i;
	uint64_t num_dat = (1ULL) << (31);
	uint64_t found_pairs_num = 0;
	uint8_t* active_val_ptr = (uint8_t*)&dat_idx;
	uint64_t key_can;
	uint8_t temp1, temp2, temp3, temp4;
	int pair_flag = 0;

	// KeyExpansion()
	RoundKey[0] = _mm_loadu_si128((const __m128i*) mk);
	RoundKey[1] = AES_128_key_exp(RoundKey[0], 0x01);
	RoundKey[2] = AES_128_key_exp(RoundKey[1], 0x02);
	RoundKey[3] = AES_128_key_exp(RoundKey[2], 0x04);
	RoundKey[4] = AES_128_key_exp(RoundKey[3], 0x08);
	RoundKey[5] = AES_128_key_exp(RoundKey[4], 0x10);
	RoundKey[6] = AES_128_key_exp(RoundKey[5], 0x20);
	RoundKey[7] = AES_128_key_exp(RoundKey[6], 0x40);
	RoundKey[8] = AES_128_key_exp(RoundKey[7], 0x80);
	RoundKey[9] = AES_128_key_exp(RoundKey[8], 0x1B);
	RoundKey[10] = AES_128_key_exp(RoundKey[9], 0x36);

	dat_tab = (pair_t*)malloc(sizeof(pair_t) * num_dat);

	srand(time(NULL));

	in_active_indexes[0] = 1;
	in_active_indexes[1] = 6;
	in_active_indexes[2] = 11;
	in_active_indexes[3] = 12;

	for (dat_idx = 0; dat_idx < num_dat; dat_idx++)
	{
		//consider little-endian
		for (idx = 0; idx < num_active_sboxes_in; idx++) state[in_active_indexes[idx]] = active_val_ptr[idx];

		///////////////////////////////////////////////////////////////
		//Encryption
		tmp_state = _mm_loadu_si128((__m128i*) state);
		tmp_state = _mm_xor_si128(tmp_state, RoundKey[0]);
		for (round_idx = 1; round_idx < round; ++round_idx)	tmp_state = _mm_aesenc_si128(tmp_state, RoundKey[round_idx]);
		tmp_state = _mm_aesenclast_si128(tmp_state, RoundKey[round]);
		///////////////////////////////////////////////////////////////

		dat_tab[dat_idx].pt = (uint8_t*)malloc(sizeof(uint8_t) * num_active_sboxes_in);
		memcpy(dat_tab[dat_idx].pt, &dat_idx, sizeof(uint8_t) * num_active_sboxes_in);
		_mm_storeu_si128((__m128i*) (dat_tab[dat_idx].ct), tmp_state);
	}

	//printf("Data Generation Finished\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 1;
	ou_pasive_indexes[5] = 4;
	ou_pasive_indexes[6] = 11;
	ou_pasive_indexes[7] = 14;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 1 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 1\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 2 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 2\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 3 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 3\n");

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 4 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 4\n", found_pairs_num);

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 5 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 5\n", found_pairs_num);

	ou_pasive_indexes[0] = 2;
	ou_pasive_indexes[1] = 5;
	ou_pasive_indexes[2] = 8;
	ou_pasive_indexes[3] = 15;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 6 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 6\n", found_pairs_num);

	// 할당했던 모든 pt 메모리 해제
	for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
		free(dat_tab[dat_idx].pt);
	}

	free(dat_tab);

	//printf("##########################################################\n");

	if (pair_flag == 0) return NO_PAIRS;

	return SUCC;
}

uint64_t MUL_OF_8_KEY_RECOVERY_THIRD_DIAGONAL(uint8_t mk[16], uint8_t state[16], int32_t round)
{
	uint8_t round_idx = 0;
	__m128i RoundKey[AES128_128_ROUND + 1];
	__m128i tmp_state;
	pair_t* dat_tab = NULL;
	uint64_t dat_idx;
	uint64_t dat = 0, a, cnt, pair_cnt, cnt_idx;
	uint64_t pair_idx[2];
	int idx;
	int i;
	uint64_t num_dat = (1ULL) << (31);
	uint64_t found_pairs_num = 0;
	uint8_t* active_val_ptr = (uint8_t*)&dat_idx;
	uint64_t key_can;
	uint8_t temp1, temp2, temp3, temp4;
	int pair_flag = 0;

	// KeyExpansion()
	RoundKey[0] = _mm_loadu_si128((const __m128i*) mk);
	RoundKey[1] = AES_128_key_exp(RoundKey[0], 0x01);
	RoundKey[2] = AES_128_key_exp(RoundKey[1], 0x02);
	RoundKey[3] = AES_128_key_exp(RoundKey[2], 0x04);
	RoundKey[4] = AES_128_key_exp(RoundKey[3], 0x08);
	RoundKey[5] = AES_128_key_exp(RoundKey[4], 0x10);
	RoundKey[6] = AES_128_key_exp(RoundKey[5], 0x20);
	RoundKey[7] = AES_128_key_exp(RoundKey[6], 0x40);
	RoundKey[8] = AES_128_key_exp(RoundKey[7], 0x80);
	RoundKey[9] = AES_128_key_exp(RoundKey[8], 0x1B);
	RoundKey[10] = AES_128_key_exp(RoundKey[9], 0x36);

	dat_tab = (pair_t*)malloc(sizeof(pair_t) * num_dat);

	srand(time(NULL));

	in_active_indexes[0] = 2;
	in_active_indexes[1] = 7;
	in_active_indexes[2] = 8;
	in_active_indexes[3] = 13;

	for (dat_idx = 0; dat_idx < num_dat; dat_idx++)
	{
		//consider little-endian
		for (idx = 0; idx < num_active_sboxes_in; idx++) state[in_active_indexes[idx]] = active_val_ptr[idx];

		///////////////////////////////////////////////////////////////
		//Encryption
		tmp_state = _mm_loadu_si128((__m128i*) state);
		tmp_state = _mm_xor_si128(tmp_state, RoundKey[0]);
		for (round_idx = 1; round_idx < round; ++round_idx)	tmp_state = _mm_aesenc_si128(tmp_state, RoundKey[round_idx]);
		tmp_state = _mm_aesenclast_si128(tmp_state, RoundKey[round]);
		///////////////////////////////////////////////////////////////

		dat_tab[dat_idx].pt = (uint8_t*)malloc(sizeof(uint8_t) * num_active_sboxes_in);
		memcpy(dat_tab[dat_idx].pt, &dat_idx, sizeof(uint8_t) * num_active_sboxes_in);
		_mm_storeu_si128((__m128i*) (dat_tab[dat_idx].ct), tmp_state);
	}

	//printf("Data Generation Finished\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 1;
	ou_pasive_indexes[5] = 4;
	ou_pasive_indexes[6] = 11;
	ou_pasive_indexes[7] = 14;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 1 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 1\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 2 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 2\n");

	ou_pasive_indexes[0] = 0;
	ou_pasive_indexes[1] = 7;
	ou_pasive_indexes[2] = 10;
	ou_pasive_indexes[3] = 13;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 3 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally Pairs are Found in Data 3\n");

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 2;
	ou_pasive_indexes[5] = 5;
	ou_pasive_indexes[6] = 8;
	ou_pasive_indexes[7] = 15;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 4 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 4\n", found_pairs_num);

	ou_pasive_indexes[0] = 1;
	ou_pasive_indexes[1] = 4;
	ou_pasive_indexes[2] = 11;
	ou_pasive_indexes[3] = 14;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 5 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 5\n", found_pairs_num);

	ou_pasive_indexes[0] = 2;
	ou_pasive_indexes[1] = 5;
	ou_pasive_indexes[2] = 8;
	ou_pasive_indexes[3] = 15;
	ou_pasive_indexes[4] = 3;
	ou_pasive_indexes[5] = 6;
	ou_pasive_indexes[6] = 9;
	ou_pasive_indexes[7] = 12;

	qsort(dat_tab, num_dat, sizeof(pair_t), comp_states);

	//printf("Sorting Data 6 Finished\n");

	pair_cnt = 0;

	for (dat_idx = 1; dat_idx < num_dat; dat_idx++)
	{
		if (comp_states(&(dat_tab[(dat_idx - 1)]), &(dat_tab[dat_idx])) == 0)
		{
			pair_idx[pair_cnt] = dat_idx;

			pair_cnt += 1;

			if (pair_cnt == 2)
			{
				pair_flag = 1;
				for (i = 0; i < 4; i++)
				{
					for (key_can = 0; key_can < 256; key_can++)
					{
						temp1 = dat_tab[pair_idx[0] - 1].pt[i] ^ key_can;
						temp2 = dat_tab[pair_idx[0]].pt[i] ^ key_can;
						temp1 = sbox[temp1];
						temp2 = sbox[temp2];
						temp3 = dat_tab[pair_idx[1] - 1].pt[i] ^ key_can;
						temp4 = dat_tab[pair_idx[1]].pt[i] ^ key_can;
						temp3 = sbox[temp3];
						temp4 = sbox[temp4];
						if ((temp1 ^ temp2) == (temp3 ^ temp4))
						{
							if (key_can != mk[in_active_indexes[i]])
							{
								for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
									free(dat_tab[dat_idx].pt);
								}

								free(dat_tab);
								return FAIL;
							}
							break;
						}
					}
				}
				for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
					free(dat_tab[dat_idx].pt);
				}

				free(dat_tab);
				return SUCC;
			}
		}
	}

	//printf("Totally %llu Pairs are Found in Data 6\n", found_pairs_num);

	// 할당했던 모든 pt 메모리 해제
	for (dat_idx = 0; dat_idx < num_dat; dat_idx++) {
		free(dat_tab[dat_idx].pt);
	}

	free(dat_tab);

	//printf("##########################################################\n");

	if (pair_flag == 0) return NO_PAIRS;

	return SUCC;
}
