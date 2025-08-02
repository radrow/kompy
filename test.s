.text

.globl main
main:
      # ######################################################################
      # Function: main
      # ######################################################################
    
      # If condition
    
      # Compiling args for f
    li t0, 1  # Load integer 1
      # Saving registers: ['ra', 't1', 't2', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    addi sp, sp, -56
    sw ra, 0(sp)
    sw t1, 4(sp)
    sw t2, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)
    sw t5, 20(sp)
    sw t6, 24(sp)
    sw a0, 28(sp)
    sw a1, 32(sp)
    sw a2, 36(sp)
    sw a3, 40(sp)
    sw a4, 44(sp)
    sw a5, 48(sp)
    sw a6, 52(sp)
    
      # Loading args for 'f'
    mv a0, t0  # Load argument 0
    jal ra, fun$f  # Call function 'f'
    mv t0, a0  # Move return value
      # Restoring registers: ['ra', 't1', 't2', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    lw ra, 0(sp)
    lw t1, 4(sp)
    lw t2, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    lw t5, 20(sp)
    lw t6, 24(sp)
    lw a0, 28(sp)
    lw a1, 32(sp)
    lw a2, 36(sp)
    lw a3, 40(sp)
    lw a4, 44(sp)
    lw a5, 48(sp)
    lw a6, 52(sp)
    addi sp, sp, 56
    
    
      # Compiling args for f
    li t1, 2  # Load integer 2
      # Saving registers: ['ra', 't0', 't2', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    addi sp, sp, -56
    sw ra, 0(sp)
    sw t0, 4(sp)
    sw t2, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)
    sw t5, 20(sp)
    sw t6, 24(sp)
    sw a0, 28(sp)
    sw a1, 32(sp)
    sw a2, 36(sp)
    sw a3, 40(sp)
    sw a4, 44(sp)
    sw a5, 48(sp)
    sw a6, 52(sp)
    
      # Loading args for 'f'
    mv a0, t1  # Load argument 0
    jal ra, fun$f  # Call function 'f'
    mv t1, a0  # Move return value
      # Restoring registers: ['ra', 't0', 't2', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    lw ra, 0(sp)
    lw t0, 4(sp)
    lw t2, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    lw t5, 20(sp)
    lw t6, 24(sp)
    lw a0, 28(sp)
    lw a1, 32(sp)
    lw a2, 36(sp)
    lw a3, 40(sp)
    lw a4, 44(sp)
    lw a5, 48(sp)
    lw a6, 52(sp)
    addi sp, sp, 56
    
    sub t0, t0, t1  # Compute equality
    seqz t0, t0
    
      # Compiling args for f
    li t1, 2  # Load integer 2
      # Saving registers: ['ra', 't0', 't2', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    addi sp, sp, -56
    sw ra, 0(sp)
    sw t0, 4(sp)
    sw t2, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)
    sw t5, 20(sp)
    sw t6, 24(sp)
    sw a0, 28(sp)
    sw a1, 32(sp)
    sw a2, 36(sp)
    sw a3, 40(sp)
    sw a4, 44(sp)
    sw a5, 48(sp)
    sw a6, 52(sp)
    
      # Loading args for 'f'
    mv a0, t1  # Load argument 0
    jal ra, fun$f  # Call function 'f'
    mv t1, a0  # Move return value
      # Restoring registers: ['ra', 't0', 't2', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    lw ra, 0(sp)
    lw t0, 4(sp)
    lw t2, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    lw t5, 20(sp)
    lw t6, 24(sp)
    lw a0, 28(sp)
    lw a1, 32(sp)
    lw a2, 36(sp)
    lw a3, 40(sp)
    lw a4, 44(sp)
    lw a5, 48(sp)
    lw a6, 52(sp)
    addi sp, sp, 56
    
    
      # Compiling args for f
    li t2, 3  # Load integer 3
      # Saving registers: ['ra', 't0', 't1', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    addi sp, sp, -56
    sw ra, 0(sp)
    sw t0, 4(sp)
    sw t1, 8(sp)
    sw t3, 12(sp)
    sw t4, 16(sp)
    sw t5, 20(sp)
    sw t6, 24(sp)
    sw a0, 28(sp)
    sw a1, 32(sp)
    sw a2, 36(sp)
    sw a3, 40(sp)
    sw a4, 44(sp)
    sw a5, 48(sp)
    sw a6, 52(sp)
    
      # Loading args for 'f'
    mv a0, t2  # Load argument 0
    jal ra, fun$f  # Call function 'f'
    mv t2, a0  # Move return value
      # Restoring registers: ['ra', 't0', 't1', 't3', 't4', 't5', 't6', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    lw ra, 0(sp)
    lw t0, 4(sp)
    lw t1, 8(sp)
    lw t3, 12(sp)
    lw t4, 16(sp)
    lw t5, 20(sp)
    lw t6, 24(sp)
    lw a0, 28(sp)
    lw a1, 32(sp)
    lw a2, 36(sp)
    lw a3, 40(sp)
    lw a4, 44(sp)
    lw a5, 48(sp)
    lw a6, 52(sp)
    addi sp, sp, 56
    
    sub t1, t1, t2  # Compute equality
    seqz t1, t1
    sub t0, t0, t1  # Compute equality
    seqz t0, t0
    beqz t0, main$_if_end_0  # If-branch
main$_if_end_0:
    li a0, 0  # Exit with code 0
    li a7, 10
    ecall


fun$f:
      # ######################################################################
      # Function: f
      # ######################################################################
    
      # Saving registers: ['sp', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
    addi sp, sp, -52
    sw sp, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    sw s4, 20(sp)
    sw s5, 24(sp)
    sw s6, 28(sp)
    sw s7, 32(sp)
    sw s8, 36(sp)
    sw s9, 40(sp)
    sw s10, 44(sp)
    sw s11, 48(sp)
    
    mv s0, a0  # Save parameter 'x'
    mv t0, s0  # Load variable 'x'
    mv a0, t0  # Print integer
    li a7, 1
    ecall
    mv t0, s0  # Load variable 'x'
    mv a0, t0  # Move return value
      # Restoring registers: ['sp', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
    lw sp, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    lw s4, 20(sp)
    lw s5, 24(sp)
    lw s6, 28(sp)
    lw s7, 32(sp)
    lw s8, 36(sp)
    lw s9, 40(sp)
    lw s10, 44(sp)
    lw s11, 48(sp)
    addi sp, sp, 52
    
    ret  # Return from function
