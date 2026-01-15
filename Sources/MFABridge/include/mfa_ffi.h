#ifndef MFA_FFI_H
#define MFA_FFI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/**
 * @brief Error codes for MFA operations
 *
 * All functions return 0 on success, positive values on error.
 */
typedef enum {
    MFA_SUCCESS = 0,                        ///< Operation completed successfully
    MFA_ERROR_INVALID_ARGS = 1,            ///< Invalid arguments provided
    MFA_ERROR_MEMORY_ALLOCATION = 2,       ///< Memory allocation failed
    MFA_ERROR_DEVICE_NOT_SUPPORTED = 3,    ///< Metal device not available
    MFA_ERROR_KERNEL_COMPILATION = 4,      ///< Metal kernel compilation failed
    MFA_ERROR_EXECUTION_FAILED = 5         ///< Kernel execution failed
} mfa_error_t;

/**
 * @brief Precision types for tensor operations
 *
 * Controls the precision used for different stages of computation.
 */
typedef enum {
    MFA_PRECISION_FP16 = 0,                ///< Half precision (16-bit float)
    MFA_PRECISION_BF16 = 1,                ///< BFloat16 (16-bit)
    MFA_PRECISION_FP32 = 2,                ///< Single precision (32-bit float)
    MFA_PRECISION_INT8 = 3,
    MFA_PRECISION_INT4 = 4
} mfa_precision_t;

/* ... other declarations for buffers, context, etc. ... */

/**
 * @brief Check if Metal is supported on this device
 *
 * @return true if Metal device is available, false otherwise
 */
bool mfa_is_device_supported(void);

/**
 * @brief Check if the current Metal device supports native BFloat16 (bf16)
 *
 * Returns true if the current system/GPU supports native BF16 arithmetic/textures/etc.
 * This allows higher layers to decide whether to use BF16 kernels, or fall back to FP32/emulation.
 */
bool mfa_is_bfloat16_supported(void);

/**
 * @brief Get the version of the MFA library
 *
 * @param[out] major Major version number
 * @param[out] minor Minor version number
 * @param[out] patch Patch version number
 */
void mfa_get_version(int* major, int* minor, int* patch);

/* ... other prototypes ... */

#ifdef __cplusplus
}
#endif

#endif // MFA_FFI_H
