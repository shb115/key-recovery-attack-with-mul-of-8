#ifndef _MULOF8_H_
#define _MULOF8_H_


#ifdef __cplusplus
extern "C" {
#endif


#ifndef NOCRYPT
#define NOCRYPT
#endif

#if defined _MSC_VER
	//Visual Studio
#ifdef _DEVELOPMENT
#define DEV_DEFINE __declspec(dllexport)
#else
#define DEV_DEFINE __declspec(dllimport)
#endif
#elif defined __GNUC__
	//GCC
#ifdef _DEVELOPMENT
#define DEV_DEFINE __attribute__ ((visibility("default")))
#else
	//nothing to define
#define DEV_DEFINE 
#endif
#endif

#if defined	__NO_INLINE__
#define DEV_INLINE //nothing
#else
#define DEV_INLINE inline
#endif

#include <stdint.h>
#include <stdio.h>

	//ciphers
	DEV_DEFINE int AES128_128_ENC	(uint8_t ct[16], uint8_t pt[16], uint8_t mk[16], int32_t round);
	DEV_DEFINE int AES128_128_DEC	(uint8_t pt[16], uint8_t ct[16], uint8_t mk[16], int32_t round);
	DEV_DEFINE int AES128_192_ENC	(uint8_t ct[16], uint8_t pt[16], uint8_t mk[24], int32_t round);
	DEV_DEFINE int AES128_192_DEC	(uint8_t pt[16], uint8_t ct[16], uint8_t mk[24], int32_t round);
	DEV_DEFINE int AES128_256_ENC	(uint8_t ct[16], uint8_t pt[16], uint8_t mk[32], int32_t round);
	DEV_DEFINE int AES128_256_DEC	(uint8_t pt[16], uint8_t ct[16], uint8_t mk[32], int32_t round);

	//analysis
	DEV_DEFINE void MUL_OF_8_DISTINGUISHER_FOUND_PAIRS(uint8_t mk[16], uint8_t state[16], int32_t round);
	DEV_DEFINE uint64_t MUL_OF_8_DISTINGUISHER_NUM_ONLY(uint8_t mk[16], uint8_t state[16], int32_t round);
	DEV_DEFINE uint64_t MUL_OF_8_KEY_RECOVERY_FIRST_DIAGONAL(uint8_t mk[16], uint8_t state[16], int32_t round);
	DEV_DEFINE uint64_t MUL_OF_8_KEY_RECOVERY_SECOND_DIAGONAL(uint8_t mk[16], uint8_t state[16], int32_t round);
	DEV_DEFINE uint64_t MUL_OF_8_KEY_RECOVERY_THIRD_DIAGONAL(uint8_t mk[16], uint8_t state[16], int32_t round);

#ifdef __cplusplus
}
#endif /*extern "C"*/

#endif /*_MULOF8_H_*/