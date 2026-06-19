/**
 * @file    DeepGemm_W8A8_I8_PERCHANNEL_ASM_TN_MT256X256X128_FP16.s
 */

// Component.Signature.SignatureCOV3
.amdgcn_target "amdgcn-amd-amdhsa--gfx938"
.text
	.protected DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE
	.globl DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE
	.p2align 8
	.type DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE,@function
	.section .rodata,#alloc
	.p2align 6
	.amdhsa_kernel DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE
	  .amdhsa_user_sgpr_kernarg_segment_ptr 1
	  .amdhsa_next_free_vgpr 255 // vgprs
	  .amdhsa_next_free_sgpr 102 // sgprs
	  .amdhsa_group_segment_fixed_size 65536 // lds bytes
	  .amdhsa_private_segment_fixed_size 0
	  .amdhsa_system_sgpr_workgroup_id_x 1
	  .amdhsa_system_sgpr_workgroup_id_y 1
	  .amdhsa_system_sgpr_workgroup_id_z 1
	  .amdhsa_system_vgpr_workitem_id 0
     .amdhsa_float_denorm_mode_32 3
     .amdhsa_float_denorm_mode_16_64 3
	.end_amdhsa_kernel
.text

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE
    .symbol: 'DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            SizesFree0
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          4
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
      - .name:            D
        .size:            8
        .offset:          16
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          24
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          48
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          52
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      f32
      - .name:            AddressScaleA
        .size:            8
        .offset:          88
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleB
        .size:            8
        .offset:          96
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleAlphaVec
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            bias
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            biasType
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            StrideBias
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            activationAlpha
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationBeta
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationType
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   65536
    .kernarg_segment_align:      8
    .kernarg_segment_size:       144
    .max_flat_workgroup_size:    768
    .private_segment_fixed_size: 0
    .sgpr_count:                 102
    .sgpr_spill_count:           0
    .vgpr_count:                 255
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE:

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
.macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst:req, src0:req, src1:req, dpp=
   v_add_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_i32 dst:req, src0:req, src1:req, dpp=
   v_add_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=
   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=
   v_sub_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_i32 dst:req, src0:req, src1:req, dpp=
   v_sub_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro K3_ADDR_ADD_COL lo:req, hi:req
   _v_add_co_u32 \lo, vcc, \lo, v134
   v_addc_co_u32 \hi, vcc, \hi, v144, vcc
.endm

.macro K3_ADDR_INC lo:req, hi:req
   _v_add_co_u32 \lo, vcc, \lo, 0x20
   v_addc_co_u32 \hi, vcc, \hi, v144, vcc
.endm

.macro K3_TABLE_ADDR_INC bytes:req
   v_mov_b32 v147, \bytes
   _v_add_co_u32 v145, vcc, v145, v147
   v_addc_co_u32 v146, vcc, v146, v144, vcc
.endm

.macro K3_LOAD_COMBINE_ADDR4 off0:req, off1:req, off2:req, off3:req
   v_mov_b32 v144, 0
   v_lshlrev_b32 v134, 1, v128

   v_add_u32 v145, \off0, v129
   v_lshlrev_b32 v145, 3, v145
   v_mov_b32 v146, s93
   _v_add_co_u32 v145, vcc, v145, s92
   v_addc_co_u32 v146, vcc, v146, v144, vcc
   global_load_dwordx2 v[136:137], v[145:146], off

   v_add_u32 v145, \off1, v129
   v_lshlrev_b32 v145, 3, v145
   v_mov_b32 v146, s93
   _v_add_co_u32 v145, vcc, v145, s92
   v_addc_co_u32 v146, vcc, v146, v144, vcc
   global_load_dwordx2 v[138:139], v[145:146], off

   v_add_u32 v145, \off2, v129
   v_lshlrev_b32 v145, 3, v145
   v_mov_b32 v146, s93
   _v_add_co_u32 v145, vcc, v145, s92
   v_addc_co_u32 v146, vcc, v146, v144, vcc
   global_load_dwordx2 v[140:141], v[145:146], off

   v_add_u32 v145, \off3, v129
   v_lshlrev_b32 v145, 3, v145
   v_mov_b32 v146, s93
   _v_add_co_u32 v145, vcc, v145, s92
   v_addc_co_u32 v146, vcc, v146, v144, vcc
   global_load_dwordx2 v[142:143], v[145:146], off

   s_waitcnt vmcnt(0)
   v_or_b32 v152, v136, v137
   v_or_b32 v153, v138, v139
   v_or_b32 v154, v140, v141
   v_or_b32 v155, v142, v143
   K3_ADDR_ADD_COL v136, v137
   K3_ADDR_ADD_COL v138, v139
   K3_ADDR_ADD_COL v140, v141
   K3_ADDR_ADD_COL v142, v143
.endm

.macro K3_STORE4 data0:req, data1:req, data2:req, data3:req
   global_store_short v[136:137], \data0, off
   global_store_short v[138:139], \data1, off
   global_store_short v[140:141], \data2, off
   global_store_short v[142:143], \data3, off
.endm

.macro K3_PACK8_BF16 data:req
   v_lshrrev_b32 v148, 6, v[vgprSerial]
   v_lshlrev_b32 v148, 7, v148
   v_lshlrev_b32 v149, 5, v133
   v_add_u32 v148, v148, v149
   v_lshrrev_b32 v149, 3, v132
   v_lshlrev_b32 v149, 4, v149
   v_add_u32 v148, v148, v149
   v_and_b32 v149, 7, v132
   v_lshlrev_b32 v149, 1, v149
   v_add_u32 v149, v148, v149
   ds_write_b16 v149, \data
   s_waitcnt lgkmcnt(0)
   ds_read_b128 v[232:235], v148
   s_waitcnt lgkmcnt(0)
.endm

.macro K3_STORE_VEC8 addr:req, valid:req
   v_and_b32 v148, 7, v132
   v_cmp_eq_u32 vcc, 0, v148
   s_and_saveexec_b64 s[82:83], vcc
   s_cbranch_execz .L_k3_store_vec8_done_\@
   v_cmp_ne_u32 vcc, 0, \valid
   s_and_saveexec_b64 s[84:85], vcc
   s_cbranch_execz .L_k3_store_vec8_skip_\@
   global_store_dwordx4 \addr, v[232:235], off
.L_k3_store_vec8_skip_\@:
   s_mov_b64 exec, s[84:85]
.L_k3_store_vec8_done_\@:
   s_mov_b64 exec, s[82:83]
.endm

.macro K3_STORE_C4 data0:req, data1:req, data2:req, data3:req
   K3_PACK8_BF16 \data0
   K3_STORE_VEC8 v[136:137], v152
   K3_PACK8_BF16 \data1
   K3_STORE_VEC8 v[138:139], v153
   K3_PACK8_BF16 \data2
   K3_STORE_VEC8 v[140:141], v154
   K3_PACK8_BF16 \data3
   K3_STORE_VEC8 v[142:143], v155
.endm

.macro K3_STAGE_ONE_H0 data:req, rowoff:req, colbytes:req
   v_add_u32 v148, \rowoff, v147
   v_lshlrev_b32 v148, 9, v148
   v_add_u32 v148, \colbytes, v148
   v_add_u32 v148, v148, v200
   ds_write_b16 v148, \data
.endm

.macro K3_STAGE_ONE_H1 data:req, rowoff:req, colbytes:req
   v_add_u32 v148, \rowoff, v147
   v_sub_u32 v148, v148, v201
   v_lshlrev_b32 v148, 9, v148
   v_add_u32 v148, \colbytes, v148
   v_add_u32 v148, v148, v200
   ds_write_b16 v148, \data
.endm

.macro K3_STAGE_C4_H0 data0:req, data1:req, data2:req, data3:req, rowoff0:req, rowoff1:req, rowoff2:req, rowoff3:req, colbytes:req
   K3_STAGE_ONE_H0 \data0, \rowoff0, \colbytes
   K3_STAGE_ONE_H0 \data1, \rowoff1, \colbytes
   K3_STAGE_ONE_H0 \data2, \rowoff2, \colbytes
   K3_STAGE_ONE_H0 \data3, \rowoff3, \colbytes
.endm

.macro K3_STAGE_C4_H1 data0:req, data1:req, data2:req, data3:req, rowoff0:req, rowoff1:req, rowoff2:req, rowoff3:req, colbytes:req
   K3_STAGE_ONE_H1 \data0, \rowoff0, \colbytes
   K3_STAGE_ONE_H1 \data1, \rowoff1, \colbytes
   K3_STAGE_ONE_H1 \data2, \rowoff2, \colbytes
   K3_STAGE_ONE_H1 \data3, \rowoff3, \colbytes
.endm

.macro K3_STAGE_TILE_H0
   K3_STAGE_C4_H0 v0, v1, v2, v3, 0, 4, 8, 12, 0
   K3_STAGE_C4_H0 v4, v5, v6, v7, 0, 4, 8, 12, 32
   K3_STAGE_C4_H0 v8, v9, v10, v11, 0, 4, 8, 12, 64
   K3_STAGE_C4_H0 v12, v13, v14, v15, 0, 4, 8, 12, 96
   K3_STAGE_C4_H0 v16, v17, v18, v19, 0, 4, 8, 12, 128
   K3_STAGE_C4_H0 v20, v21, v22, v23, 0, 4, 8, 12, 160
   K3_STAGE_C4_H0 v24, v25, v26, v27, 0, 4, 8, 12, 192
   K3_STAGE_C4_H0 v28, v29, v30, v31, 0, 4, 8, 12, 224
   K3_STAGE_C4_H0 v32, v33, v34, v35, 0, 4, 8, 12, 256
   K3_STAGE_C4_H0 v36, v37, v38, v39, 0, 4, 8, 12, 288
   K3_STAGE_C4_H0 v40, v41, v42, v43, 0, 4, 8, 12, 320
   K3_STAGE_C4_H0 v44, v45, v46, v47, 0, 4, 8, 12, 352
   K3_STAGE_C4_H0 v48, v49, v50, v51, 0, 4, 8, 12, 384
   K3_STAGE_C4_H0 v52, v53, v54, v55, 0, 4, 8, 12, 416
   K3_STAGE_C4_H0 v56, v57, v58, v59, 0, 4, 8, 12, 448
   K3_STAGE_C4_H0 v60, v61, v62, v63, 0, 4, 8, 12, 480
   K3_STAGE_C4_H0 v64, v65, v66, v67, 16, 20, 24, 28, 0
   K3_STAGE_C4_H0 v68, v69, v70, v71, 16, 20, 24, 28, 32
   K3_STAGE_C4_H0 v72, v73, v74, v75, 16, 20, 24, 28, 64
   K3_STAGE_C4_H0 v76, v77, v78, v79, 16, 20, 24, 28, 96
   K3_STAGE_C4_H0 v80, v81, v82, v83, 16, 20, 24, 28, 128
   K3_STAGE_C4_H0 v84, v85, v86, v87, 16, 20, 24, 28, 160
   K3_STAGE_C4_H0 v88, v89, v90, v91, 16, 20, 24, 28, 192
   K3_STAGE_C4_H0 v92, v93, v94, v95, 16, 20, 24, 28, 224
   K3_STAGE_C4_H0 v96, v97, v98, v99, 16, 20, 24, 28, 256
   K3_STAGE_C4_H0 v100, v101, v102, v103, 16, 20, 24, 28, 288
   K3_STAGE_C4_H0 v104, v105, v106, v107, 16, 20, 24, 28, 320
   K3_STAGE_C4_H0 v108, v109, v110, v111, 16, 20, 24, 28, 352
   K3_STAGE_C4_H0 v112, v113, v114, v115, 16, 20, 24, 28, 384
   K3_STAGE_C4_H0 v116, v117, v118, v119, 16, 20, 24, 28, 416
   K3_STAGE_C4_H0 v120, v121, v122, v123, 16, 20, 24, 28, 448
   K3_STAGE_C4_H0 v124, v125, v126, v127, 16, 20, 24, 28, 480
.endm

.macro K3_STAGE_TILE_H1
   K3_STAGE_C4_H1 v0, v1, v2, v3, 0, 4, 8, 12, 0
   K3_STAGE_C4_H1 v4, v5, v6, v7, 0, 4, 8, 12, 32
   K3_STAGE_C4_H1 v8, v9, v10, v11, 0, 4, 8, 12, 64
   K3_STAGE_C4_H1 v12, v13, v14, v15, 0, 4, 8, 12, 96
   K3_STAGE_C4_H1 v16, v17, v18, v19, 0, 4, 8, 12, 128
   K3_STAGE_C4_H1 v20, v21, v22, v23, 0, 4, 8, 12, 160
   K3_STAGE_C4_H1 v24, v25, v26, v27, 0, 4, 8, 12, 192
   K3_STAGE_C4_H1 v28, v29, v30, v31, 0, 4, 8, 12, 224
   K3_STAGE_C4_H1 v32, v33, v34, v35, 0, 4, 8, 12, 256
   K3_STAGE_C4_H1 v36, v37, v38, v39, 0, 4, 8, 12, 288
   K3_STAGE_C4_H1 v40, v41, v42, v43, 0, 4, 8, 12, 320
   K3_STAGE_C4_H1 v44, v45, v46, v47, 0, 4, 8, 12, 352
   K3_STAGE_C4_H1 v48, v49, v50, v51, 0, 4, 8, 12, 384
   K3_STAGE_C4_H1 v52, v53, v54, v55, 0, 4, 8, 12, 416
   K3_STAGE_C4_H1 v56, v57, v58, v59, 0, 4, 8, 12, 448
   K3_STAGE_C4_H1 v60, v61, v62, v63, 0, 4, 8, 12, 480
   K3_STAGE_C4_H1 v64, v65, v66, v67, 16, 20, 24, 28, 0
   K3_STAGE_C4_H1 v68, v69, v70, v71, 16, 20, 24, 28, 32
   K3_STAGE_C4_H1 v72, v73, v74, v75, 16, 20, 24, 28, 64
   K3_STAGE_C4_H1 v76, v77, v78, v79, 16, 20, 24, 28, 96
   K3_STAGE_C4_H1 v80, v81, v82, v83, 16, 20, 24, 28, 128
   K3_STAGE_C4_H1 v84, v85, v86, v87, 16, 20, 24, 28, 160
   K3_STAGE_C4_H1 v88, v89, v90, v91, 16, 20, 24, 28, 192
   K3_STAGE_C4_H1 v92, v93, v94, v95, 16, 20, 24, 28, 224
   K3_STAGE_C4_H1 v96, v97, v98, v99, 16, 20, 24, 28, 256
   K3_STAGE_C4_H1 v100, v101, v102, v103, 16, 20, 24, 28, 288
   K3_STAGE_C4_H1 v104, v105, v106, v107, 16, 20, 24, 28, 320
   K3_STAGE_C4_H1 v108, v109, v110, v111, 16, 20, 24, 28, 352
   K3_STAGE_C4_H1 v112, v113, v114, v115, 16, 20, 24, 28, 384
   K3_STAGE_C4_H1 v116, v117, v118, v119, 16, 20, 24, 28, 416
   K3_STAGE_C4_H1 v120, v121, v122, v123, 16, 20, 24, 28, 448
   K3_STAGE_C4_H1 v124, v125, v126, v127, 16, 20, 24, 28, 480
.endm

.macro K3_STORE_STAGED_HALF rowptr_offset:req, step:req
   v_mov_b32 v149, 4096
.L_k3_staged_loop_\@:
   v_cmp_lt_u32 vcc, v150, v149
   s_and_saveexec_b64 s[80:81], vcc
   s_cbranch_execz .L_k3_staged_done_\@

   v_lshrrev_b32 v151, 5, v150
   _v_add_co_u32 v152, vcc, v151, s56
   v_lshlrev_b32 v153, 3, v152
   buffer_load_dwordx2 v[136:137], v153, s[92:95], 0, offen, offset:\rowptr_offset

   v_add_u32 v157, 8, v151
   _v_add_co_u32 v158, vcc, v157, s56
   v_lshlrev_b32 v159, 3, v158
   buffer_load_dwordx2 v[138:139], v159, s[92:95], 0, offen, offset:\rowptr_offset

   v_lshlrev_b32 v148, 9, v151
   v_and_b32 v154, 31, v150
   v_lshlrev_b32 v154, 4, v154
   v_add_u32 v148, v148, v154
   ds_read_b128 v[232:235], v148

   v_lshlrev_b32 v161, 9, v157
   v_add_u32 v161, v161, v154
   ds_read_b128 v[236:239], v161

   s_waitcnt vmcnt(0)
   s_waitcnt lgkmcnt(0)
   v_or_b32 v155, v136, v137
   v_or_b32 v161, v138, v139
   v_add_u32 v162, v154, s54
   v_mov_b32 v156, 0
   _v_add_co_u32 v136, vcc, v136, v162
   v_addc_co_u32 v137, vcc, v137, v156, vcc
   _v_add_co_u32 v138, vcc, v138, v162
   v_addc_co_u32 v139, vcc, v139, v156, vcc
   v_cmp_ne_u32 vcc, 0, v155
   s_and_saveexec_b64 s[82:83], vcc
   global_store_dwordx4 v[136:137], v[232:235], off
   s_mov_b64 exec, s[82:83]
   v_cmp_ne_u32 vcc, 0, v161
   s_and_saveexec_b64 s[82:83], vcc
   global_store_dwordx4 v[138:139], v[236:239], off
   s_mov_b64 exec, s[82:83]
   s_mov_b64 exec, s[80:81]
   v_add_u32 v150, \step, v150
   s_branch .L_k3_staged_loop_\@

.L_k3_staged_done_\@:
   s_mov_b64 exec, s[80:81]
.endm

.macro K3_TAIL_ATOMIC_SIGNAL offset:req, value:req
   v_mov_b32 v153, s56
   v_mov_b32 v154, s57
   v_mov_b32 v156, \offset
   _v_add_co_u32 v153, vcc, v153, v156
   v_mov_b32 v156, 0
   v_addc_co_u32 v154, vcc, v154, v156, vcc
   global_load_dwordx2 v[136:137], v[153:154], off glc
   s_waitcnt vmcnt(0)
   v_or_b32 v155, v136, v137
   v_cmp_ne_u32 vcc, 0, v155
   s_and_saveexec_b64 s[66:67], vcc
   s_cbranch_execz .L_k3_tail_atomic_skip_\@
   v_mov_b32 v238, \value
   global_atomic_umax v240, v[136:137], v238, off glc
.L_k3_tail_atomic_skip_\@:
   s_mov_b64 exec, s[66:67]
.endm

.macro K3_TAIL_WAIT_SIGNAL offset:req
   v_mov_b32 v153, s56
   v_mov_b32 v154, s57
   v_mov_b32 v156, \offset
   _v_add_co_u32 v153, vcc, v153, v156
   v_mov_b32 v156, 0
   v_addc_co_u32 v154, vcc, v154, v156, vcc
   global_load_dwordx2 v[136:137], v[153:154], off glc
   s_waitcnt vmcnt(0)
   v_or_b32 v155, v136, v137
   v_cmp_ne_u32 vcc, 0, v155
   s_and_saveexec_b64 s[66:67], vcc
   s_cbranch_execz .L_k3_tail_wait_skip_\@
.L_k3_tail_wait_loop_\@:
   global_load_dword v238, v[136:137], off glc slc
   s_waitcnt vmcnt(0)
   buffer_wbinvl1_vol
   v_cmp_lt_u32 vcc, v238, s78
   s_cbranch_vccnz .L_k3_tail_wait_loop_\@
.L_k3_tail_wait_skip_\@:
   s_mov_b64 exec, s[66:67]
.endm

.macro K3_TAIL_ADDR_FROM_BASE base_lo:req, base_hi:req
   v_lshlrev_b32 v153, 4, v250
   v_mov_b32 v154, 0
   v_mov_b32 v156, \base_lo
   _v_add_co_u32 v153, vcc, v153, v156
   v_mov_b32 v156, \base_hi
   v_addc_co_u32 v154, vcc, v154, v156, vcc
.endm

.macro K3_TAIL_ADDR_ADD_SLOT_STRIDE
   v_mov_b32 v156, s77
   _v_add_co_u32 v153, vcc, v153, v156
   v_mov_b32 v156, 0
   v_addc_co_u32 v154, vcc, v154, v156, vcc
.endm

.macro K3_TAIL_ACCUM_DWORD packed:req, sum_lo:req, sum_hi:req
   v_and_b32 v160, 0xffff, \packed
   v_lshlrev_b32 v160, 16, v160
   v_lshrrev_b32 v161, 16, \packed
   v_lshlrev_b32 v161, 16, v161
   v_add_f32 \sum_lo, \sum_lo, v160
   v_add_f32 \sum_hi, \sum_hi, v161
.endm

.macro K3_TAIL_ACCUM_VEC d0:req, d1:req, d2:req, d3:req
   K3_TAIL_ACCUM_DWORD \d0, v180, v181
   K3_TAIL_ACCUM_DWORD \d1, v182, v183
   K3_TAIL_ACCUM_DWORD \d2, v184, v185
   K3_TAIL_ACCUM_DWORD \d3, v186, v187
.endm

.macro K3_TAIL_LOAD_ACCUM_SLOT
   global_load_dwordx4 v[232:235], v[153:154], off
   s_waitcnt vmcnt(0)
   K3_TAIL_ACCUM_VEC v232, v233, v234, v235
   K3_TAIL_ADDR_ADD_SLOT_STRIDE
.endm

.macro K3_TAIL_LOAD_ACCUM_SIX
   global_load_dwordx4 v[200:203], v[153:154], off
   K3_TAIL_ADDR_ADD_SLOT_STRIDE
   global_load_dwordx4 v[204:207], v[153:154], off
   K3_TAIL_ADDR_ADD_SLOT_STRIDE
   global_load_dwordx4 v[208:211], v[153:154], off
   K3_TAIL_ADDR_ADD_SLOT_STRIDE
   global_load_dwordx4 v[212:215], v[153:154], off
   K3_TAIL_ADDR_ADD_SLOT_STRIDE
   global_load_dwordx4 v[216:219], v[153:154], off
   K3_TAIL_ADDR_ADD_SLOT_STRIDE
   global_load_dwordx4 v[220:223], v[153:154], off
   K3_TAIL_ADDR_ADD_SLOT_STRIDE
   s_waitcnt vmcnt(0)
   K3_TAIL_ACCUM_VEC v200, v201, v202, v203
   K3_TAIL_ACCUM_VEC v204, v205, v206, v207
   K3_TAIL_ACCUM_VEC v208, v209, v210, v211
   K3_TAIL_ACCUM_VEC v212, v213, v214, v215
   K3_TAIL_ACCUM_VEC v216, v217, v218, v219
   K3_TAIL_ACCUM_VEC v220, v221, v222, v223
.endm

.macro K3_TAIL_APPLY_GRAPH_RUNTIME_STATE
   s_mov_b32 s89, 0
   s_cmp_eq_u64 s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0
   s_cbranch_scc1 .L_k3_graph_runtime_done_\@
   s_load_dword s86, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xc4
   s_waitcnt lgkmcnt(0)
   s_cmp_eq_u32 s86, 0
   s_cbranch_scc1 .L_k3_graph_runtime_done_\@
   s_load_dwordx2 s[90:91], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xc8
   s_waitcnt lgkmcnt(0)
   s_cmp_eq_u64 s[90:91], 0
   s_cbranch_scc1 .L_k3_graph_runtime_done_\@
   s_load_dword s88, s[90:91], 0x0
   s_add_u32 s90, s90, s86
   s_addc_u32 s91, s91, 0
   s_load_dword s76, s[90:91], 0x0
   s_waitcnt lgkmcnt(0)
   s_load_dwordx2 s[90:91], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xd0
   s_waitcnt lgkmcnt(0)
   s_cmp_eq_u64 s[90:91], 0
   s_cbranch_scc1 .L_k3_graph_signal_generation_done_\@
   s_load_dword s78, s[90:91], 0x0
   s_mov_b32 s89, 1
   s_waitcnt lgkmcnt(0)
   s_cmp_gt_u32 s78, 0
   s_cbranch_scc1 .L_k3_graph_signal_generation_done_\@
   s_mov_b32 s78, 1
.L_k3_graph_signal_generation_done_\@:
   s_cmp_gt_u32 s88, 0
   s_cbranch_scc1 .L_k3_graph_active_nonzero_\@
   s_mov_b32 s88, 1
.L_k3_graph_active_nonzero_\@:
   s_lshl_b32 s60, s88, 4
   s_lshl_b32 s76, s76, 9
.L_k3_graph_runtime_done_\@:
.endm

.macro K3_TAIL_F32_TO_BF16 out:req, value:req
   v_lshrrev_b32 \out, 16, \value
   v_and_b32 v161, 1, \out
   v_add_u32 \out, 0x7fff, \value
   v_add_u32 \out, v161, \out
   v_lshrrev_b32 \out, 16, \out
.endm

.macro K3_TAIL_PACK_REDUCE_OUT
   K3_TAIL_F32_TO_BF16 v232, v180
   K3_TAIL_F32_TO_BF16 v233, v181
   v_lshlrev_b32 v233, 16, v233
   v_or_b32 v232, v232, v233
   K3_TAIL_F32_TO_BF16 v233, v182
   K3_TAIL_F32_TO_BF16 v234, v183
   v_lshlrev_b32 v234, 16, v234
   v_or_b32 v233, v233, v234
   K3_TAIL_F32_TO_BF16 v234, v184
   K3_TAIL_F32_TO_BF16 v235, v185
   v_lshlrev_b32 v235, 16, v235
   v_or_b32 v234, v234, v235
   K3_TAIL_F32_TO_BF16 v235, v186
   K3_TAIL_F32_TO_BF16 v236, v187
   v_lshlrev_b32 v236, 16, v236
   v_or_b32 v235, v235, v236
.endm

.macro K3_INC_C_COL
   K3_ADDR_INC v136, v137
   K3_ADDR_INC v138, v139
   K3_ADDR_INC v140, v141
   K3_ADDR_INC v142, v143
.endm

.macro K3_SELECT_VEC8_LANES
.endm

.macro K3_RESTORE_ALL_LANES
.endm

.macro K3_INC_ADDR4
   K3_ADDR_INC v136, v137
   K3_ADDR_INC v138, v139
   K3_ADDR_INC v140, v141
   K3_ADDR_INC v142, v143
.endm

.macro K3_SCATTER_C_TILE_TO_COMBINE
   s_waitcnt vmcnt(0)
   buffer_wbinvl1_vol
   s_barrier

   s_mov_b32 s52, s[sgprAddressC+0]
   s_mov_b32 s53, s[sgprAddressC+1]
   s_mov_b32 s54, BufferLimit
   s_mov_b32 s55, Srd127_96
   s_mov_b64 s[56:57], s[92:93]
   s_mul_i32 s58, s[sgprWorkGroup1], 0x100
   s_mul_i32 s59, s[sgprWorkGroup0], 0x200
   s_mov_b32 s60, 8192
   v_mov_b32 v148, v[vgprSerial]

K3_Scatter_C_Tile_Loop:
   v_cmp_lt_u32 vcc, v148, s60
   s_and_saveexec_b64 s[80:81], vcc
   s_cbranch_execz K3_Scatter_C_Tile_Done

   v_lshrrev_b32 v149, 5, v148
   v_and_b32 v150, 31, v148
   v_lshlrev_b32 v151, 4, v150
   v_add_u32 v151, s59, v151
   v_add_u32 v152, s58, v149

   v_lshlrev_b32 v153, 13, v152
   v_add_u32 v153, v153, v151
   buffer_load_dwordx4 v[232:235], v153, s[52:55], 0, offen, offset:0

   v_lshlrev_b32 v154, 3, v152
   v_mov_b32 v155, s57
   _v_add_co_u32 v154, vcc, v154, s56
   v_addc_co_u32 v155, vcc, v155, 0, vcc
   global_load_dwordx2 v[156:157], v[154:155], off
   s_waitcnt vmcnt(0)
   v_mov_b32 v155, 0
   _v_add_co_u32 v156, vcc, v156, v151
   v_addc_co_u32 v157, vcc, v157, v155, vcc
   global_store_dwordx4 v[156:157], v[232:235], off

   s_mov_b64 exec, s[80:81]
   v_add_u32 v148, 0x300, v148
   s_branch K3_Scatter_C_Tile_Loop

K3_Scatter_C_Tile_Done:
   s_mov_b64 exec, s[80:81]
   s_waitcnt vmcnt(0)
   s_barrier
.endm

.macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_or_b32 dst:req, src0:req, shiftCnt:req, src1:req
    v_lshl_or_b32 \dst, \src0, \shiftCnt, \src1
.endm

.macro _v_cmpx_lt_i16 dst, src0, src1=
   v_cmpx_lt_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_i32 dst, src0, src1=
   v_cmpx_lt_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_i64 dst, src0, src1=
   v_cmpx_lt_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u16 dst, src0, src1=
   v_cmpx_lt_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u32 dst, src0, src1=
   v_cmpx_lt_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u64 dst, src0, src1=
   v_cmpx_lt_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i16 dst, src0, src1=
   v_cmpx_eq_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i32 dst, src0, src1=
   v_cmpx_eq_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i64 dst, src0, src1=
   v_cmpx_eq_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u16 dst, src0, src1=
   v_cmpx_eq_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u32 dst, src0, src1=
   v_cmpx_eq_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u64 dst, src0, src1=
   v_cmpx_eq_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i16 dst, src0, src1=
   v_cmpx_le_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i32 dst, src0, src1=
   v_cmpx_le_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i64 dst, src0, src1=
   v_cmpx_le_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u16 dst, src0, src1=
   v_cmpx_le_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u32 dst, src0, src1=
   v_cmpx_le_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u64 dst, src0, src1=
   v_cmpx_le_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i16 dst, src0, src1=
   v_cmpx_gt_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i32 dst, src0, src1=
   v_cmpx_gt_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i64 dst, src0, src1=
   v_cmpx_gt_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u16 dst, src0, src1=
   v_cmpx_gt_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u32 dst, src0, src1=
   v_cmpx_gt_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u64 dst, src0, src1=
   v_cmpx_gt_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i16 dst, src0, src1=
   v_cmpx_ne_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i32 dst, src0, src1=
   v_cmpx_ne_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i64 dst, src0, src1=
   v_cmpx_ne_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u16 dst, src0, src1=
   v_cmpx_ne_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u32 dst, src0, src1=
   v_cmpx_ne_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u64 dst, src0, src1=
   v_cmpx_ne_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i16 dst, src0, src1=
   v_cmpx_lg_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i32 dst, src0, src1=
   v_cmpx_lg_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i64 dst, src0, src1=
   v_cmpx_lg_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u16 dst, src0, src1=
   v_cmpx_lg_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u32 dst, src0, src1=
   v_cmpx_lg_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u64 dst, src0, src1=
   v_cmpx_lg_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i16 dst, src0, src1=
   v_cmpx_ge_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i32 dst, src0, src1=
   v_cmpx_ge_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i64 dst, src0, src1=
   v_cmpx_ge_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u16 dst, src0, src1=
   v_cmpx_ge_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u32 dst, src0, src1=
   v_cmpx_ge_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u64 dst, src0, src1=
   v_cmpx_ge_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i16 dst, src0, src1=
   v_cmpx_o_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i32 dst, src0, src1=
   v_cmpx_o_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i64 dst, src0, src1=
   v_cmpx_o_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u16 dst, src0, src1=
   v_cmpx_o_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u32 dst, src0, src1=
   v_cmpx_o_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u64 dst, src0, src1=
   v_cmpx_o_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i16 dst, src0, src1=
   v_cmpx_u_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i32 dst, src0, src1=
   v_cmpx_u_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i64 dst, src0, src1=
   v_cmpx_u_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u16 dst, src0, src1=
   v_cmpx_u_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u32 dst, src0, src1=
   v_cmpx_u_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u64 dst, src0, src1=
   v_cmpx_u_u64 \dst, \src0, \src1 
.endm
.macro _v_mac_f32 c:req, a:req, b:req
    v_mac_f32 \c, \a, \b
.endmacro


/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-128), ,  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 128
.set vgprValuB_X0_I0, 192
.set vgprValuB_X1_I0, 208
.set vgprGlobalReadOffsetA, 224
.set vgprGlobalReadOffsetB, 232
.set vgprLocalWriteAddrA, 240
.set vgprLocalWriteAddrB, 241
.set vgprLocalReadAddrA, 242
.set vgprLocalReadAddrA_ori, 243
.set vgprLocalReadAddrASwap, 244
.set vgprLocalReadAddrASwap_ori, 245
.set vgprSerial,246
.set vgprSerial_forLdsWrapOptA, 247
.set vgprG2LA, 128
.set vgprG2LB, 160
.set vgprLocalReadAddrB, 248
.set vgprLocalReadAddrASwap1, 249
.set vgprGlobalReadOffsetBForSWTL, 250
.set vgprLocalReadAddrA_forLdsWrapTailLoop, 247
.set vgprPack5OffsetBaseA, 250
.set vgprPack5OffsetTmpA, 251

.set vgprValuBias_M, 225
/* Num VGPR=240 */
/* Num AccVGPR=0 */

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprExternalArgAddress, 6
.set sgprLoopCounterL, 5
.set sgprOrigLoopCounter, 8
.set sgprSrdA, 44
.set sgprSrdB, 48
.set sgprSrdD, 12
.set sgprSrdC, 16
.set sgprNumWorkGroups0, 9
.set sgprNumWorkGroups1, 10

.set sgprSizesFree, 20
.set sgprSizesSum, 23
.set sgprAddressD, 24
.set sgprAddressC, 26
.set sgprAddressA, 28
.set sgprAddressB, 30
.set sgprAlpha, 40
.set sgprBeta, 41
.set sgprStridesD, 32
.set sgprStridesC, 34
.set sgprStridesA, 36
.set sgprStridesB, 38

.set sgprOrigStaggerUIter, 50
.set sgprNumFullBlocks, 53
.set sgprWgmRemainder1, 54
.set sgprMagicNumberWgmRemainder1, 55
.set sgprOffsetD, 56
.set sgprOffsetC, 57
.set sgprOffsetA, 58
.set sgprOffsetB, 59
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprGlobalReadIncsA, 101
.set sgprGlobalReadIncsB, 100
.set sgprLocalWriteAddrA, 60
.set sgprLocalWriteAddrB, 61
.set sgprMask, 72
.set sgprWaveiD, 73
.set sgprStrideSwizzleB, 74
.set sgprStructNumB, 75
.set sgprStructBitB, 76
.set sgprStructBit1B, 77
.set sgprStrideStructB, 78
.set sgprOffsetStructB, 79
.set sgprLocalWriteAddrA_ori, 80
.set sgprLoopforPf, 81
.set sgprShadowLimitB_forTailLoop, 82
.set sgprShadowLimitA_forTailLoop, 88
.set sgprSrdA_forTailLoop, 84

.set sgprGSU, 42
.set sgprScaleA, 98
.set sgprScaleB, 99
/* max SGPR=102 */
.set sgprScaleFlag, 96
.set sgprSrdscaleA, 44
.set sgprSrdscaleB, 48
/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1
.set sgprStrideoffC, sgprSrdA
.set sgprStrideoffD, sgprAddressD
.set sgprEdgeCheck, sgprAddressB

.set MT0, 256
.set MT1, 256
.set DepthU, 128
.set BpeA, 1
.set BpeALog2, 0
.set BpeB, 1
.set BpeBLog2, 0
.set BpeAGR, 1
.set BpeAGRLog2, 0
.set BpeBGR, 1
.set BpeBGRLog2, 0

/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 16
.set SrdShiftLeftB, 16
/* 4GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xfffffffe
.set BufferOOB, 0xfffffffe
.set Srd127_96, 0x00020000


.macro COMPUTE_SACLE  vgprValueX:req vgprValueZ:req vgprValueY:req vgprValueW:req
	v_pk_mul_f32 v[\vgprValueX+0:\vgprValueX+1], v[\vgprValueZ+0:\vgprValueZ+1], v[\vgprValueY+0:\vgprValueY+1]
	v_pk_mul_f32 v[\vgprValueX+2:\vgprValueX+3], v[\vgprValueZ+0:\vgprValueZ+1], v[\vgprValueY+2:\vgprValueY+3]
	v_pk_mul_f32 v[\vgprValueX+4:\vgprValueX+5], v[\vgprValueZ+2:\vgprValueZ+3], v[\vgprValueY+0:\vgprValueY+1]
	v_pk_mul_f32 v[\vgprValueX+6:\vgprValueX+7], v[\vgprValueZ+2:\vgprValueZ+3], v[\vgprValueY+2:\vgprValueY+3]

	v_pk_mul_f32 v[\vgprValueX+8:\vgprValueX+9], v[\vgprValueZ+4:\vgprValueZ+5], v[\vgprValueY+0:\vgprValueY+1]
	v_pk_mul_f32 v[\vgprValueX+10:\vgprValueX+11], v[\vgprValueZ+4:\vgprValueZ+5], v[\vgprValueY+2:\vgprValueY+3]
	v_pk_mul_f32 v[\vgprValueX+12:\vgprValueX+13], v[\vgprValueZ+6:\vgprValueZ+7], v[\vgprValueY+0:\vgprValueY+1]
	v_pk_mul_f32 v[\vgprValueX+14:\vgprValueX+15], v[\vgprValueZ+6:\vgprValueZ+7], v[\vgprValueY+2:\vgprValueY+3]
	
	v_pk_mul_f32 v[\vgprValueW+0:\vgprValueW+1], v[\vgprValueX+0:\vgprValueX+1], v[\vgprValueW+0:\vgprValueW+1]
	v_pk_mul_f32 v[\vgprValueW+2:\vgprValueW+3], v[\vgprValueX+2:\vgprValueX+3], v[\vgprValueW+2:\vgprValueW+3]
	v_pk_mul_f32 v[\vgprValueW+4:\vgprValueW+5], v[\vgprValueX+4:\vgprValueX+5], v[\vgprValueW+4:\vgprValueW+5]
	v_pk_mul_f32 v[\vgprValueW+6:\vgprValueW+7], v[\vgprValueX+6:\vgprValueX+7], v[\vgprValueW+6:\vgprValueW+7]

	v_pk_mul_f32 v[\vgprValueW+8:\vgprValueW+9], v[\vgprValueX+8:\vgprValueX+9],  v[\vgprValueW+8:\vgprValueW+9]
	v_pk_mul_f32 v[\vgprValueW+10:\vgprValueW+11], v[\vgprValueX+10:\vgprValueX+11],  v[\vgprValueW+10:\vgprValueW+11]
	v_pk_mul_f32 v[\vgprValueW+12:\vgprValueW+13], v[\vgprValueX+12:\vgprValueX+13], v[\vgprValueW+12:\vgprValueW+13]
	v_pk_mul_f32 v[\vgprValueW+14:\vgprValueW+15], v[\vgprValueX+14:\vgprValueX+15],  v[\vgprValueW+14:\vgprValueW+15]
	
.endm

.macro FMA_ADD_VALUEC_0
	COMPUTE_SACLE 180 136 168 0
	COMPUTE_SACLE 180 144 168 16
	COMPUTE_SACLE 180 152 168 32
	COMPUTE_SACLE 180 160 168 48
.endm

.macro FMA_ADD_VALUEC_64
	COMPUTE_SACLE 180 136 172 64
	COMPUTE_SACLE 180 144 172 80
	COMPUTE_SACLE 180 152 172 96
	COMPUTE_SACLE 180 160 172 112
.endm

.macro COMPUTE_ADDRESS_SCALE
	v_and_b32 v250, 255, v[vgprSerial]
	v_lshlrev_b32 v250, 2, v250
	v_and_b32 v251, 255, v[vgprSerial]
	v_lshlrev_b32 v251, 2, v251

	v_lshlrev_b32 v252, 2, v[vgprSerial]
	v_readfirstlane_b32 s[sgprLocalWriteAddrA], v252

   /***************scale A  - E_id***********/
   s_mul_hi_u32   s65, s[sgprSizeI], s[sgprScaleFlag]
   s_mul_i32      s64, s[sgprSizeI], s[sgprScaleFlag]
   /*****************************************/
   s_lshl_b64 s[64:65], s[64:65], 2

	s_mul_hi_u32 s67, 4*MT0, s[sgprWorkGroup0]
	s_mul_i32 s66, 4*MT0, s[sgprWorkGroup0]

   s_add_u32 s64, s64, s66
   s_addc_u32 s65, s65, s67

	s_add_u32 s[sgprSrdA], s92, s64
	s_addc_u32 s[sgprSrdA+1], s93, s65
	s_lshl_b32 s[sgprSrdA+2], s[sgprSizeI], 2
   s_sub_u32  s[sgprSrdA+2], s[sgprSrdA+2], s66
	s_mov_b32 s[sgprSrdA+3], Srd127_96

   s_mul_hi_u32 s65, s[sgprSizeJ], s[sgprWorkGroup2]
   s_mul_i32    s64, s[sgprSizeJ], s[sgprWorkGroup2]
   s_lshl_b64 s[64:65], s[64:65], 2

	s_mul_hi_u32 s67, 4*MT1, s[sgprWorkGroup1]
	s_mul_i32 s66, 4*MT1, s[sgprWorkGroup1]

   s_add_u32 s64, s64, s66
   s_addc_u32 s65, s65, s67

	s_add_u32 s[sgprSrdB], s94, s64
	s_addc_u32 s[sgprSrdB+1], s95,s65
	s_lshl_b32 s[sgprSrdB+2], s[sgprSizeJ], 2
   s_sub_u32  s[sgprSrdB+2], s[sgprSrdB+2], s66
	s_mov_b32 s[sgprSrdB+3], Srd127_96

   /************这里要设置index src***********/
   s_add_u32 s52, s98,  s64
   s_addc_u32 s53, s99, s65
   s_lshl_b32 s54, s[sgprSizeJ], 2
   s_sub_u32  s54, s54, s66
   s_mov_b32 s55, Srd127_96
   /****************************************/

	v_and_b32 v252, 63, v[vgprSerial]
	v_lshrrev_b32 v253, 6, v[vgprSerial]
	v_and_b32 v253, 7, v253

	v_and_b32 v248, 15, v252
	v_lshrrev_b32 v249, 4, v252

	v_and_b32 v245, 0, v253
	v_lshrrev_b32 v247, 0, v253

	v_lshlrev_b32 v245, 5, v245
	v_lshlrev_b32 v247, 5, v247

	v_add_lshl_u32 v253, v248, v245, 0x2
	v_add_lshl_u32 v252, v249, v247, 0x2
	v_add_co_u32 v252, vcc, 0x1000, v252
   v_add_co_u32 v254, vcc, 0x1000, v252
.endm

.macro SMQUANT
	COMPUTE_ADDRESS_SCALE
	
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 m0, s[sgprLocalWriteAddrA]
	buffer_load_dword v250, s[sgprSrdA:sgprSrdA+3], 0, offen, offset:0x00, lds
	s_add_u32 m0, m0, 0x1000
	buffer_load_dword v251, s[sgprSrdB:sgprSrdB+3], 0, offen, offset:0x00, lds

   /*************index***************************************/
   s_add_u32 m0, m0, 0x1000
   buffer_load_dword v251, s[52:55], 0, offen, offset:0x00, lds
   /********************************************************/

	s_waitcnt vmcnt(0)
	s_barrier

	ds_read_b32 v136, v253, offset:0x00
	ds_read_b32 v138, v253, offset:0x40
	ds_read_b32 v140, v253, offset:0x80
	ds_read_b32 v142, v253, offset:0xc0
	ds_read_b32 v144, v253, offset:0x100
	ds_read_b32 v146, v253, offset:0x140
	ds_read_b32 v148, v253, offset:0x180
	ds_read_b32 v150, v253, offset:0x1c0
	ds_read_b32 v152, v253, offset:0x200
	ds_read_b32 v154, v253, offset:0x240
	ds_read_b32 v156, v253, offset:0x280
	ds_read_b32 v158, v253, offset:0x2c0
	ds_read_b32 v160, v253, offset:0x300
	ds_read_b32 v162, v253, offset:0x340
	ds_read_b32 v164, v253, offset:0x380
	ds_read_b32 v166, v253, offset:0x3c0


	ds_read_b32 v168, v252, offset:0x00
	ds_read_b32 v169, v252, offset:0x10
	ds_read_b32 v170, v252, offset:0x20
	ds_read_b32 v171, v252, offset:0x30
	ds_read_b32 v172, v252, offset:0x40
	ds_read_b32 v173, v252, offset:0x50
	ds_read_b32 v174, v252, offset:0x60
	ds_read_b32 v175, v252, offset:0x70

   /***********读取index************/
   ds_read_b32 v176, v254, offset:0x00
   ds_read_b32 v177, v254, offset:0x10
   ds_read_b32 v178, v254, offset:0x20
   ds_read_b32 v179, v254, offset:0x30
   ds_read_b32 v180, v254, offset:0x40
   ds_read_b32 v181, v254, offset:0x50
   ds_read_b32 v182, v254, offset:0x60
   ds_read_b32 v183, v254, offset:0x70
   
	s_waitcnt lgkmcnt(0)
	s_barrier

   /********判断置零*****************/
   v_cmp_eq_u32 s[50:51], v176, -1
   v_cndmask_b32 v168, v168, 0, s[50:51]
   v_cmp_eq_u32 s[50:51], v177, -1
   v_cndmask_b32 v169, v169, 0, s[50:51]
   v_cmp_eq_u32 s[50:51], v178, -1
   v_cndmask_b32 v170, v170, 0, s[50:51]
   v_cmp_eq_u32 s[50:51], v179, -1
   v_cndmask_b32 v171, v171, 0, s[50:51]
   v_cmp_eq_u32 s[50:51], v180, -1
   v_cndmask_b32 v172, v172, 0, s[50:51]
   v_cmp_eq_u32 s[50:51], v181, -1
   v_cndmask_b32 v173, v173, 0, s[50:51]
   v_cmp_eq_u32 s[50:51], v182, -1
   v_cndmask_b32 v174, v174, 0, s[50:51]
   v_cmp_eq_u32 s[50:51], v183, -1
   v_cndmask_b32 v175, v175, 0, s[50:51]
   /*******************************/
	v_mov_b32 v137, v136
	v_mov_b32 v139, v138
	v_mov_b32 v141, v140
	v_mov_b32 v143, v142
	v_mov_b32 v145, v144
	v_mov_b32 v147, v146
	v_mov_b32 v149, v148
	v_mov_b32 v151, v150
	
	v_mov_b32 v153, v152
	v_mov_b32 v155, v154
	v_mov_b32 v157, v156
	v_mov_b32 v159, v158
	v_mov_b32 v161, v160
	v_mov_b32 v163, v162
	v_mov_b32 v165, v164
	v_mov_b32 v167, v166

   FMA_ADD_VALUEC_0
   FMA_ADD_VALUEC_64

.endm

.macro FP32_TO_BF16 vgprValue:req
  v_cmp_u_f32 s[98:99], v[\vgprValue], v[\vgprValue] // check Nan
  v_bfe_u32 v253, v[\vgprValue], 16, 1               // Non-Nan case: store lsb of bf16
  v_add3_u32 v253, v[\vgprValue], v253, v252           // Non-Nan case: add lsb and the increment
  v_cndmask_b32 v[\vgprValue], v253, v251, s[98:99]   //
  v_lshrrev_b32 v[\vgprValue], 16, v[\vgprValue]   // convert C to bf16'
   ; v_cvt_bf16_f32 v[\vgprValue], v[\vgprValue]
.endm

.macro FP32_TO_BF16_ALL vgprValue:req
  v_mov_b32 v251, 0x7fff0000     // fp32 Nan
  v_mov_b32 v252, 0x7fff        // rounding bias for bfloat16
  FP32_TO_BF16 \vgprValue+0
  FP32_TO_BF16 \vgprValue+1
  FP32_TO_BF16 \vgprValue+2
  FP32_TO_BF16 \vgprValue+3
  FP32_TO_BF16 \vgprValue+4
  FP32_TO_BF16 \vgprValue+5
  FP32_TO_BF16 \vgprValue+6
  FP32_TO_BF16 \vgprValue+7
  FP32_TO_BF16 \vgprValue+8
  FP32_TO_BF16 \vgprValue+9
  FP32_TO_BF16 \vgprValue+10
  FP32_TO_BF16 \vgprValue+11
  FP32_TO_BF16 \vgprValue+12
  FP32_TO_BF16 \vgprValue+13
  FP32_TO_BF16 \vgprValue+14
  FP32_TO_BF16 \vgprValue+15
  FP32_TO_BF16 \vgprValue+16
  FP32_TO_BF16 \vgprValue+17
  FP32_TO_BF16 \vgprValue+18
  FP32_TO_BF16 \vgprValue+19
  FP32_TO_BF16 \vgprValue+20
  FP32_TO_BF16 \vgprValue+21
  FP32_TO_BF16 \vgprValue+22
  FP32_TO_BF16 \vgprValue+23
  FP32_TO_BF16 \vgprValue+24
  FP32_TO_BF16 \vgprValue+25
  FP32_TO_BF16 \vgprValue+26
  FP32_TO_BF16 \vgprValue+27
  FP32_TO_BF16 \vgprValue+28
  FP32_TO_BF16 \vgprValue+29
  FP32_TO_BF16 \vgprValue+30
  FP32_TO_BF16 \vgprValue+31
  FP32_TO_BF16 \vgprValue+32
  FP32_TO_BF16 \vgprValue+33
  FP32_TO_BF16 \vgprValue+34
  FP32_TO_BF16 \vgprValue+35
  FP32_TO_BF16 \vgprValue+36
  FP32_TO_BF16 \vgprValue+37
  FP32_TO_BF16 \vgprValue+38
  FP32_TO_BF16 \vgprValue+39
  FP32_TO_BF16 \vgprValue+40
  FP32_TO_BF16 \vgprValue+41
  FP32_TO_BF16 \vgprValue+42
  FP32_TO_BF16 \vgprValue+43
  FP32_TO_BF16 \vgprValue+44
  FP32_TO_BF16 \vgprValue+45
  FP32_TO_BF16 \vgprValue+46
  FP32_TO_BF16 \vgprValue+47
  FP32_TO_BF16 \vgprValue+48
  FP32_TO_BF16 \vgprValue+49
  FP32_TO_BF16 \vgprValue+50
  FP32_TO_BF16 \vgprValue+51
  FP32_TO_BF16 \vgprValue+52
  FP32_TO_BF16 \vgprValue+53
  FP32_TO_BF16 \vgprValue+54
  FP32_TO_BF16 \vgprValue+55
  FP32_TO_BF16 \vgprValue+56
  FP32_TO_BF16 \vgprValue+57
  FP32_TO_BF16 \vgprValue+58
  FP32_TO_BF16 \vgprValue+59
  FP32_TO_BF16 \vgprValue+60
  FP32_TO_BF16 \vgprValue+61
  FP32_TO_BF16 \vgprValue+62
  FP32_TO_BF16 \vgprValue+63
  FP32_TO_BF16 \vgprValue+64
  FP32_TO_BF16 \vgprValue+65
  FP32_TO_BF16 \vgprValue+66
  FP32_TO_BF16 \vgprValue+67
  FP32_TO_BF16 \vgprValue+68
  FP32_TO_BF16 \vgprValue+69
  FP32_TO_BF16 \vgprValue+70
  FP32_TO_BF16 \vgprValue+71
  FP32_TO_BF16 \vgprValue+72
  FP32_TO_BF16 \vgprValue+73
  FP32_TO_BF16 \vgprValue+74
  FP32_TO_BF16 \vgprValue+75
  FP32_TO_BF16 \vgprValue+76
  FP32_TO_BF16 \vgprValue+77
  FP32_TO_BF16 \vgprValue+78
  FP32_TO_BF16 \vgprValue+79
  FP32_TO_BF16 \vgprValue+80
  FP32_TO_BF16 \vgprValue+81
  FP32_TO_BF16 \vgprValue+82
  FP32_TO_BF16 \vgprValue+83
  FP32_TO_BF16 \vgprValue+84
  FP32_TO_BF16 \vgprValue+85
  FP32_TO_BF16 \vgprValue+86
  FP32_TO_BF16 \vgprValue+87
  FP32_TO_BF16 \vgprValue+88
  FP32_TO_BF16 \vgprValue+89
  FP32_TO_BF16 \vgprValue+90
  FP32_TO_BF16 \vgprValue+91
  FP32_TO_BF16 \vgprValue+92
  FP32_TO_BF16 \vgprValue+93
  FP32_TO_BF16 \vgprValue+94
  FP32_TO_BF16 \vgprValue+95
  FP32_TO_BF16 \vgprValue+96
  FP32_TO_BF16 \vgprValue+97
  FP32_TO_BF16 \vgprValue+98
  FP32_TO_BF16 \vgprValue+99
  FP32_TO_BF16 \vgprValue+100
  FP32_TO_BF16 \vgprValue+101
  FP32_TO_BF16 \vgprValue+102
  FP32_TO_BF16 \vgprValue+103
  FP32_TO_BF16 \vgprValue+104
  FP32_TO_BF16 \vgprValue+105
  FP32_TO_BF16 \vgprValue+106
  FP32_TO_BF16 \vgprValue+107
  FP32_TO_BF16 \vgprValue+108
  FP32_TO_BF16 \vgprValue+109
  FP32_TO_BF16 \vgprValue+110
  FP32_TO_BF16 \vgprValue+111
  FP32_TO_BF16 \vgprValue+112
  FP32_TO_BF16 \vgprValue+113
  FP32_TO_BF16 \vgprValue+114
  FP32_TO_BF16 \vgprValue+115
  FP32_TO_BF16 \vgprValue+116
  FP32_TO_BF16 \vgprValue+117
  FP32_TO_BF16 \vgprValue+118
  FP32_TO_BF16 \vgprValue+119
  FP32_TO_BF16 \vgprValue+120
  FP32_TO_BF16 \vgprValue+121
  FP32_TO_BF16 \vgprValue+122
  FP32_TO_BF16 \vgprValue+123
  FP32_TO_BF16 \vgprValue+124
  FP32_TO_BF16 \vgprValue+125
  FP32_TO_BF16 \vgprValue+126
  FP32_TO_BF16 \vgprValue+127
.endm

.macro FP32_TO_BF16_7FFF
  v_mov_b32 v251, 0x7fff0000     // fp32 Nan
  v_mov_b32 v252, 0x7fff        // rounding bias for bfloat16
.endm

.macro FP32_TO_BF16_Single vgprValue:req
  FP32_TO_BF16 \vgprValue
.endm

.macro BF16_TO_FP32_Single vgprValue:req
  v_lshlrev_b32 v[\vgprValue], 16, v[\vgprValue]
.endm

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffsetL:req vgprOffset0I:req vgprTmp:req
   v_lshlrev_b32 v[\vgprTmp+0], 10, v[\vgprOffset0I] // no256 * (4096 / 4)
   _v_add_co_u32 v[\vgprAddr+0], vcc, v[vgprPack5OffsetBaseA], v[\vgprTmp+0]
   _v_add_u32 v[\vgprAddr+0], 0x10, v[\vgprAddr+0]    // keep original ASM prepad compensation
                                                      // offset *= bytes/element (multiplier is 1, do nothing)
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
   v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
   _v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
   _v_add_u32 v[\vgprAddr+0], 0x10, v[\vgprAddr+0]    // add prepad for pointer shift
                                                      // offset *= bytes/element (multiplier is 1, do nothing)
.endm

/******************************************/
/* Swizzle Struct Buffer B                */
/******************************************/
.macro GLOBAL_STRUCT_OFFSET_B vgprAddr:req vgprOffset:req vgprIndex:req
   v_mul_lo_u32 v[\vgprAddr+0], s[sgprOffsetStructB], v[\vgprIndex] // 
   v_add_u32 v[\vgprAddr+0], v[\vgprAddr+0], v[\vgprOffset] // 
   v_add_u32 v[\vgprAddr+0], 0x10, v[\vgprAddr+0]     // add prepad for pointer shift
   v_lshlrev_b32 v[\vgprAddr+0], 0x0, v[\vgprAddr+0]  // global struct offset B
.endm

.macro GLOBAL_STRUCT_INDEX_B vgprAddr:req vgprIndex:req
   v_mul_lo_u32 v[\vgprAddr+0], s[sgprStrideStructB], v[\vgprIndex+0] // global struct index B
.endm


/******************************************/
/* 16x8 thread-tile                       */
/******************************************/
.macro MAC_16x8_X0
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0:vgprValuC+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+0:vgprValuC+3] 
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4:vgprValuC+7], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+4:vgprValuC+7] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8:vgprValuC+11], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+8:vgprValuC+11] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12:vgprValuC+15], v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+12:vgprValuC+15] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16:vgprValuC+19], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+16:vgprValuC+19] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20:vgprValuC+23], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+20:vgprValuC+23] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24:vgprValuC+27], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+24:vgprValuC+27] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28:vgprValuC+31], v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprValuC+28:vgprValuC+31] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32:vgprValuC+35], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+32:vgprValuC+35] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36:vgprValuC+39], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+36:vgprValuC+39] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40:vgprValuC+43], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+40:vgprValuC+43] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44:vgprValuC+47], v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+44:vgprValuC+47] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48:vgprValuC+51], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+48:vgprValuC+51] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52:vgprValuC+55], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+52:vgprValuC+55] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56:vgprValuC+59], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+56:vgprValuC+59] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60:vgprValuC+63], v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprValuC+60:vgprValuC+63] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64:vgprValuC+67], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+64:vgprValuC+67] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68:vgprValuC+71], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+68:vgprValuC+71] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72:vgprValuC+75], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+72:vgprValuC+75] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76:vgprValuC+79], v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+76:vgprValuC+79] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80:vgprValuC+83], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+80:vgprValuC+83] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84:vgprValuC+87], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+84:vgprValuC+87] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88:vgprValuC+91], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+88:vgprValuC+91] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92:vgprValuC+95], v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprValuC+92:vgprValuC+95] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96:vgprValuC+99], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+96:vgprValuC+99] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100:vgprValuC+103], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+100:vgprValuC+103] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104:vgprValuC+107], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+104:vgprValuC+107] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108:vgprValuC+111], v[vgprValuA_X0_I0+6:vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+108:vgprValuC+111] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112:vgprValuC+115], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+112:vgprValuC+115] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116:vgprValuC+119], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+116:vgprValuC+119] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120:vgprValuC+123], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+120:vgprValuC+123] 
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124:vgprValuC+127], v[vgprValuA_X0_I0+14:vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprValuC+124:vgprValuC+127] 
s_setprio 0 // Reset priority after macs 
.endm



.macro MAC_32x4x2_X0_I0
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+0: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4: vgprValuC+4+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+4: vgprValuC+4+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+8: vgprValuC+8+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12: vgprValuC+12+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+12: vgprValuC+12+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+16: vgprValuC+16+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20: vgprValuC+20+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+20: vgprValuC+20+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+24: vgprValuC+24+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28: vgprValuC+28+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+28: vgprValuC+28+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+32: vgprValuC+32+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36: vgprValuC+36+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+36: vgprValuC+36+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+40: vgprValuC+40+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44: vgprValuC+44+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+44: vgprValuC+44+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+48: vgprValuC+48+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52: vgprValuC+52+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+52: vgprValuC+52+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+56: vgprValuC+56+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60: vgprValuC+60+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+0: vgprValuB_X0_I0+1], v[vgprValuC+60: vgprValuC+60+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+64: vgprValuC+64+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68: vgprValuC+68+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+68: vgprValuC+68+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+72: vgprValuC+72+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76: vgprValuC+76+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+76: vgprValuC+76+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+80: vgprValuC+80+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84: vgprValuC+84+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+84: vgprValuC+84+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+88: vgprValuC+88+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92: vgprValuC+92+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+92: vgprValuC+92+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+96: vgprValuC+96+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100: vgprValuC+100+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+100: vgprValuC+100+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+104: vgprValuC+104+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108: vgprValuC+108+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+108: vgprValuC+108+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+112: vgprValuC+112+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116: vgprValuC+116+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+116: vgprValuC+116+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+120: vgprValuC+120+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124: vgprValuC+124+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+8: vgprValuB_X0_I0+9], v[vgprValuC+124: vgprValuC+124+3]

v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0: vgprValuC+0+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+0: vgprValuC+0+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+4: vgprValuC+4+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8: vgprValuC+8+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+8: vgprValuC+8+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+12: vgprValuC+12+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16: vgprValuC+16+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+16: vgprValuC+16+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+20: vgprValuC+20+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24: vgprValuC+24+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+24: vgprValuC+24+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+28: vgprValuC+28+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32: vgprValuC+32+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+32: vgprValuC+32+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+36: vgprValuC+36+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40: vgprValuC+40+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+40: vgprValuC+40+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+44: vgprValuC+44+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48: vgprValuC+48+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+48: vgprValuC+48+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+52: vgprValuC+52+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56: vgprValuC+56+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+56: vgprValuC+56+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+2: vgprValuB_X0_I0+3], v[vgprValuC+60: vgprValuC+60+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64: vgprValuC+64+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+64: vgprValuC+64+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+68: vgprValuC+68+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72: vgprValuC+72+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+72: vgprValuC+72+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+76: vgprValuC+76+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80: vgprValuC+80+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+80: vgprValuC+80+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+84: vgprValuC+84+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88: vgprValuC+88+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+88: vgprValuC+88+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+92: vgprValuC+92+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96: vgprValuC+96+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+96: vgprValuC+96+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+100: vgprValuC+100+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104: vgprValuC+104+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+104: vgprValuC+104+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+108: vgprValuC+108+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112: vgprValuC+112+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+112: vgprValuC+112+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+116: vgprValuC+116+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120: vgprValuC+120+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+120: vgprValuC+120+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+10: vgprValuB_X0_I0+11], v[vgprValuC+124: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm

.macro MAC_32x4x2_X0_I1
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+0: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4: vgprValuC+4+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+4: vgprValuC+4+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+8: vgprValuC+8+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12: vgprValuC+12+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+12: vgprValuC+12+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+16: vgprValuC+16+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20: vgprValuC+20+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+20: vgprValuC+20+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+24: vgprValuC+24+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28: vgprValuC+28+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+28: vgprValuC+28+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+32: vgprValuC+32+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36: vgprValuC+36+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+36: vgprValuC+36+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+40: vgprValuC+40+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44: vgprValuC+44+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+44: vgprValuC+44+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+48: vgprValuC+48+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52: vgprValuC+52+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+52: vgprValuC+52+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+56: vgprValuC+56+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60: vgprValuC+60+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+4: vgprValuB_X0_I0+5], v[vgprValuC+60: vgprValuC+60+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+64: vgprValuC+64+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68: vgprValuC+68+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+68: vgprValuC+68+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+72: vgprValuC+72+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76: vgprValuC+76+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+76: vgprValuC+76+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+80: vgprValuC+80+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84: vgprValuC+84+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+84: vgprValuC+84+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+88: vgprValuC+88+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92: vgprValuC+92+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+92: vgprValuC+92+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+96: vgprValuC+96+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100: vgprValuC+100+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+100: vgprValuC+100+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+104: vgprValuC+104+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108: vgprValuC+108+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+108: vgprValuC+108+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+112: vgprValuC+112+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116: vgprValuC+116+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+116: vgprValuC+116+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+120: vgprValuC+120+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124: vgprValuC+124+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X0_I0+12: vgprValuB_X0_I0+13], v[vgprValuC+124: vgprValuC+124+3]

v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0: vgprValuC+0+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+0: vgprValuC+0+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+4: vgprValuC+4+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8: vgprValuC+8+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+8: vgprValuC+8+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+12: vgprValuC+12+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16: vgprValuC+16+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+16: vgprValuC+16+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+20: vgprValuC+20+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24: vgprValuC+24+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+24: vgprValuC+24+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+28: vgprValuC+28+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32: vgprValuC+32+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+32: vgprValuC+32+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+36: vgprValuC+36+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40: vgprValuC+40+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+40: vgprValuC+40+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+44: vgprValuC+44+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48: vgprValuC+48+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+48: vgprValuC+48+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+52: vgprValuC+52+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56: vgprValuC+56+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+56: vgprValuC+56+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+6: vgprValuB_X0_I0+7], v[vgprValuC+60: vgprValuC+60+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64: vgprValuC+64+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+64: vgprValuC+64+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+68: vgprValuC+68+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72: vgprValuC+72+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+72: vgprValuC+72+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+76: vgprValuC+76+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80: vgprValuC+80+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+80: vgprValuC+80+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+84: vgprValuC+84+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88: vgprValuC+88+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+88: vgprValuC+88+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+92: vgprValuC+92+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96: vgprValuC+96+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+96: vgprValuC+96+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+100: vgprValuC+100+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104: vgprValuC+104+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+104: vgprValuC+104+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+108: vgprValuC+108+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112: vgprValuC+112+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+112: vgprValuC+112+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+116: vgprValuC+116+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120: vgprValuC+120+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+120: vgprValuC+120+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X0_I0+14: vgprValuB_X0_I0+15], v[vgprValuC+124: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm
.macro MAC_32x4x2_X1_I0
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+0: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4: vgprValuC+4+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+4: vgprValuC+4+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+8: vgprValuC+8+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12: vgprValuC+12+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+12: vgprValuC+12+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+16: vgprValuC+16+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20: vgprValuC+20+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+20: vgprValuC+20+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+24: vgprValuC+24+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28: vgprValuC+28+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+28: vgprValuC+28+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+32: vgprValuC+32+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36: vgprValuC+36+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+36: vgprValuC+36+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+40: vgprValuC+40+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44: vgprValuC+44+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+44: vgprValuC+44+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+48: vgprValuC+48+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52: vgprValuC+52+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+52: vgprValuC+52+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+56: vgprValuC+56+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60: vgprValuC+60+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+0: vgprValuB_X1_I0+1], v[vgprValuC+60: vgprValuC+60+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+64: vgprValuC+64+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68: vgprValuC+68+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+68: vgprValuC+68+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+72: vgprValuC+72+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76: vgprValuC+76+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+76: vgprValuC+76+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+80: vgprValuC+80+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84: vgprValuC+84+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+84: vgprValuC+84+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+88: vgprValuC+88+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92: vgprValuC+92+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+92: vgprValuC+92+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+96: vgprValuC+96+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100: vgprValuC+100+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+100: vgprValuC+100+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+104: vgprValuC+104+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108: vgprValuC+108+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+108: vgprValuC+108+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+112: vgprValuC+112+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116: vgprValuC+116+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+116: vgprValuC+116+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+120: vgprValuC+120+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124: vgprValuC+124+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+8: vgprValuB_X1_I0+9], v[vgprValuC+124: vgprValuC+124+3]

v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0: vgprValuC+0+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+0: vgprValuC+0+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+4: vgprValuC+4+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8: vgprValuC+8+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+8: vgprValuC+8+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+12: vgprValuC+12+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16: vgprValuC+16+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+16: vgprValuC+16+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+20: vgprValuC+20+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24: vgprValuC+24+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+24: vgprValuC+24+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+28: vgprValuC+28+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32: vgprValuC+32+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+32: vgprValuC+32+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+36: vgprValuC+36+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40: vgprValuC+40+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+40: vgprValuC+40+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+44: vgprValuC+44+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48: vgprValuC+48+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+48: vgprValuC+48+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+52: vgprValuC+52+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56: vgprValuC+56+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+56: vgprValuC+56+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+2: vgprValuB_X1_I0+3], v[vgprValuC+60: vgprValuC+60+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64: vgprValuC+64+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+64: vgprValuC+64+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+68: vgprValuC+68+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72: vgprValuC+72+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+72: vgprValuC+72+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+76: vgprValuC+76+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80: vgprValuC+80+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+80: vgprValuC+80+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+84: vgprValuC+84+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88: vgprValuC+88+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+88: vgprValuC+88+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+92: vgprValuC+92+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96: vgprValuC+96+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+96: vgprValuC+96+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+100: vgprValuC+100+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104: vgprValuC+104+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+104: vgprValuC+104+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+108: vgprValuC+108+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112: vgprValuC+112+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+112: vgprValuC+112+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+116: vgprValuC+116+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120: vgprValuC+120+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+120: vgprValuC+120+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+10: vgprValuB_X1_I0+11], v[vgprValuC+124: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm
.macro MAC_32x4x2_X1_I1
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0: vgprValuC+0+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+0: vgprValuC+0+3]
s_setprio 1 // Raise priority while processing macs
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4: vgprValuC+4+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+4: vgprValuC+4+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8: vgprValuC+8+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+8: vgprValuC+8+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12: vgprValuC+12+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+12: vgprValuC+12+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16: vgprValuC+16+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+16: vgprValuC+16+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20: vgprValuC+20+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+20: vgprValuC+20+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24: vgprValuC+24+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+24: vgprValuC+24+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28: vgprValuC+28+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+28: vgprValuC+28+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32: vgprValuC+32+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+32: vgprValuC+32+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36: vgprValuC+36+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+36: vgprValuC+36+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40: vgprValuC+40+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+40: vgprValuC+40+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44: vgprValuC+44+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+44: vgprValuC+44+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48: vgprValuC+48+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+48: vgprValuC+48+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52: vgprValuC+52+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+52: vgprValuC+52+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56: vgprValuC+56+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+56: vgprValuC+56+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60: vgprValuC+60+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+4: vgprValuB_X1_I0+5], v[vgprValuC+60: vgprValuC+60+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64: vgprValuC+64+3], v[vgprValuA_X0_I0+0: vgprValuA_X0_I0+1], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+64: vgprValuC+64+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68: vgprValuC+68+3], v[vgprValuA_X0_I0+4: vgprValuA_X0_I0+5], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+68: vgprValuC+68+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72: vgprValuC+72+3], v[vgprValuA_X0_I0+8: vgprValuA_X0_I0+9], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+72: vgprValuC+72+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76: vgprValuC+76+3], v[vgprValuA_X0_I0+12: vgprValuA_X0_I0+13], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+76: vgprValuC+76+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80: vgprValuC+80+3], v[vgprValuA_X0_I0+16: vgprValuA_X0_I0+17], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+80: vgprValuC+80+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84: vgprValuC+84+3], v[vgprValuA_X0_I0+20: vgprValuA_X0_I0+21], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+84: vgprValuC+84+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88: vgprValuC+88+3], v[vgprValuA_X0_I0+24: vgprValuA_X0_I0+25], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+88: vgprValuC+88+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92: vgprValuC+92+3], v[vgprValuA_X0_I0+28: vgprValuA_X0_I0+29], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+92: vgprValuC+92+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96: vgprValuC+96+3], v[vgprValuA_X0_I0+32: vgprValuA_X0_I0+33], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+96: vgprValuC+96+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100: vgprValuC+100+3], v[vgprValuA_X0_I0+36: vgprValuA_X0_I0+37], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+100: vgprValuC+100+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104: vgprValuC+104+3], v[vgprValuA_X0_I0+40: vgprValuA_X0_I0+41], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+104: vgprValuC+104+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108: vgprValuC+108+3], v[vgprValuA_X0_I0+44: vgprValuA_X0_I0+45], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+108: vgprValuC+108+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112: vgprValuC+112+3], v[vgprValuA_X0_I0+48: vgprValuA_X0_I0+49], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+112: vgprValuC+112+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116: vgprValuC+116+3], v[vgprValuA_X0_I0+52: vgprValuA_X0_I0+53], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+116: vgprValuC+116+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120: vgprValuC+120+3], v[vgprValuA_X0_I0+56: vgprValuA_X0_I0+57], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+120: vgprValuC+120+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124: vgprValuC+124+3], v[vgprValuA_X0_I0+60: vgprValuA_X0_I0+61], v[vgprValuB_X1_I0+12: vgprValuB_X1_I0+13], v[vgprValuC+124: vgprValuC+124+3]

v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+0: vgprValuC+0+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+0: vgprValuC+0+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+4: vgprValuC+4+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+4: vgprValuC+4+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+8: vgprValuC+8+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+8: vgprValuC+8+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+12: vgprValuC+12+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+12: vgprValuC+12+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+16: vgprValuC+16+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+16: vgprValuC+16+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+20: vgprValuC+20+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+20: vgprValuC+20+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+24: vgprValuC+24+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+24: vgprValuC+24+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+28: vgprValuC+28+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+28: vgprValuC+28+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+32: vgprValuC+32+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+32: vgprValuC+32+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+36: vgprValuC+36+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+36: vgprValuC+36+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+40: vgprValuC+40+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+40: vgprValuC+40+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+44: vgprValuC+44+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+44: vgprValuC+44+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+48: vgprValuC+48+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+48: vgprValuC+48+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+52: vgprValuC+52+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+52: vgprValuC+52+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+56: vgprValuC+56+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+56: vgprValuC+56+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+60: vgprValuC+60+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+6: vgprValuB_X1_I0+7], v[vgprValuC+60: vgprValuC+60+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+64: vgprValuC+64+3], v[vgprValuA_X0_I0+2: vgprValuA_X0_I0+3], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+64: vgprValuC+64+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+68: vgprValuC+68+3], v[vgprValuA_X0_I0+6: vgprValuA_X0_I0+7], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+68: vgprValuC+68+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+72: vgprValuC+72+3], v[vgprValuA_X0_I0+10: vgprValuA_X0_I0+11], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+72: vgprValuC+72+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+76: vgprValuC+76+3], v[vgprValuA_X0_I0+14: vgprValuA_X0_I0+15], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+76: vgprValuC+76+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+80: vgprValuC+80+3], v[vgprValuA_X0_I0+18: vgprValuA_X0_I0+19], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+80: vgprValuC+80+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+84: vgprValuC+84+3], v[vgprValuA_X0_I0+22: vgprValuA_X0_I0+23], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+84: vgprValuC+84+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+88: vgprValuC+88+3], v[vgprValuA_X0_I0+26: vgprValuA_X0_I0+27], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+88: vgprValuC+88+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+92: vgprValuC+92+3], v[vgprValuA_X0_I0+30: vgprValuA_X0_I0+31], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+92: vgprValuC+92+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+96: vgprValuC+96+3], v[vgprValuA_X0_I0+34: vgprValuA_X0_I0+35], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+96: vgprValuC+96+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+100: vgprValuC+100+3], v[vgprValuA_X0_I0+38: vgprValuA_X0_I0+39], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+100: vgprValuC+100+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+104: vgprValuC+104+3], v[vgprValuA_X0_I0+42: vgprValuA_X0_I0+43], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+104: vgprValuC+104+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+108: vgprValuC+108+3], v[vgprValuA_X0_I0+46: vgprValuA_X0_I0+47], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+108: vgprValuC+108+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+112: vgprValuC+112+3], v[vgprValuA_X0_I0+50: vgprValuA_X0_I0+51], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+112: vgprValuC+112+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+116: vgprValuC+116+3], v[vgprValuA_X0_I0+54: vgprValuA_X0_I0+55], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+116: vgprValuC+116+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+120: vgprValuC+120+3], v[vgprValuA_X0_I0+58: vgprValuA_X0_I0+59], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+120: vgprValuC+120+3]
v_mmac_f32_16x16x32_fp8_fp8 v[vgprValuC+124: vgprValuC+124+3], v[vgprValuA_X0_I0+62: vgprValuA_X0_I0+63], v[vgprValuB_X1_I0+14: vgprValuB_X1_I0+15], v[vgprValuC+124: vgprValuC+124+3]
s_setprio 0 // Reset priority after macs 
.endm


/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Grouped Gemm: Load num of Gemms */
s_load_dword s27, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0

/* Grouped Gemm: Load GSU data */
s_load_dword s[sgprGSU], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18

/* Grouped Gemm: Load address of external kernel arguments */
s_load_dwordx2 s[sgprExternalArgAddress:sgprExternalArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4

s_load_dwordx2 s[98:99], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x1c
/* Grouped Gemm: Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc
s_waitcnt lgkmcnt(0)

s_mov_b32 m0, 0x10000                              // LDS clamp at 65536 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* Check if custom structure pointer is null */
s_cmp_eq_u64 s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0 // s[ExternalArgAddress] == 0 ?
s_cbranch_scc0 label_IsExternalValid               // branch if s[ExternalArgAddress] != 0
s_mov_b32 s26, 140
s_mul_i32 s22, s27, 4
s_mov_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s26, 128
s_mov_b32 s22, 0x0
s_mov_b64 s[16:17], s[sgprExternalArgAddress:sgprExternalArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s24, 1
s_mov_b32 s23, 0
s_load_dwordx4 s[12:15], s[16:17], s22             // s12-M  s13-N   s14-B  s15-K
s_cmpk_eq_u32 s27, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s20, s12, 8                             // s20 = s12 / 256
s_and_b32 s18, 255, s12                            // s18 = s12 % 256
s_addc_u32 s20, s20, 0x0
s_lshr_b32 s21, s13, 8                             // s21 = s13 / 256
s_and_b32 s18, 255, s13                            // s18 = s13 % 256
s_addc_u32 s21, s21, 0x0
s_mul_i32 s20, s20, s21
s_mul_i32 s20, s20, s14
s_mul_i32 s20, s20, s[sgprGSU]
s_add_u32 s23, s23, s20
s_cmp_lt_u32 s[sgprWorkGroup0], s23
s_cbranch_scc1 label_FOUND
s_add_u32 s22, s22, s26
s_load_dwordx4 s[12:15], s[16:17], s22
s_add_u32 s24, s24, 1
s_cmp_lt_u32 s24, s27
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s20, s12, 8                             // s20 = s12 / 256
s_and_b32 s18, 255, s12                            // s18 = s12 % 256
s_addc_u32 s20, s20, 0x0
s_lshr_b32 s21, s13, 8                             // s21 = s13 / 256
s_and_b32 s18, 255, s13                            // s18 = s13 % 256
s_addc_u32 s21, s21, 0x0
s_mul_i32 s20, s20, s21
s_mul_i32 s20, s20, s14
s_mul_i32 s20, s20, s[sgprGSU]
s_add_u32 s23, s23, s20
s_cmp_lt_u32 s[sgprWorkGroup0], s23
s_cbranch_scc1 label_FOUND
s_sub_u32 s52, s[sgprWorkGroup0], s23

label_K3_ExtraReducerWG:
s_load_dwordx2 s[56:57], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x88
s_load_dwordx4 s[60:63], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x90
s_load_dwordx2 s[72:73], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xa0
s_load_dwordx2 s[74:75], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xa8
s_load_dwordx4 s[76:79], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xb0
s_load_dword s80, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xc0
s_waitcnt lgkmcnt(0)
K3_TAIL_APPLY_GRAPH_RUNTIME_STATE
s_cmp_eq_u32 s62, 1
s_cbranch_scc0 .L_k3_extra_reduce_endpgm
s_cmp_eq_u32 s79, 1
s_cbranch_scc0 .L_k3_extra_reduce_endpgm
s_cmp_eq_u64 s[56:57], 0
s_cbranch_scc1 .L_k3_extra_reduce_endpgm
s_cmp_eq_u64 s[72:73], 0
s_cbranch_scc1 .L_k3_extra_reduce_endpgm
s_cmp_eq_u64 s[74:75], 0
s_cbranch_scc1 .L_k3_extra_reduce_endpgm
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 .L_k3_extra_reduce_endpgm
s_cmp_ge_u32 s52, s80
s_cbranch_scc1 .L_k3_extra_reduce_endpgm

v_cmp_eq_u32 vcc, v[vgprSerial], 0
s_and_saveexec_b64 s[64:65], vcc
s_cbranch_execz .L_k3_extra_wait_done
s_cmp_gt_u32 s61, 0
s_cbranch_scc0 .L_k3_extra_wait_rank0_done
K3_TAIL_WAIT_SIGNAL 64
.L_k3_extra_wait_rank0_done:
s_cmp_gt_u32 s61, 1
s_cbranch_scc0 .L_k3_extra_wait_rank1_done
K3_TAIL_WAIT_SIGNAL 72
.L_k3_extra_wait_rank1_done:
s_cmp_gt_u32 s61, 2
s_cbranch_scc0 .L_k3_extra_wait_rank2_done
K3_TAIL_WAIT_SIGNAL 80
.L_k3_extra_wait_rank2_done:
s_cmp_gt_u32 s61, 3
s_cbranch_scc0 .L_k3_extra_wait_rank3_done
K3_TAIL_WAIT_SIGNAL 88
.L_k3_extra_wait_rank3_done:
s_cmp_gt_u32 s61, 4
s_cbranch_scc0 .L_k3_extra_wait_rank4_done
K3_TAIL_WAIT_SIGNAL 96
.L_k3_extra_wait_rank4_done:
s_cmp_gt_u32 s61, 5
s_cbranch_scc0 .L_k3_extra_wait_rank5_done
K3_TAIL_WAIT_SIGNAL 104
.L_k3_extra_wait_rank5_done:
s_cmp_gt_u32 s61, 6
s_cbranch_scc0 .L_k3_extra_wait_rank6_done
K3_TAIL_WAIT_SIGNAL 112
.L_k3_extra_wait_rank6_done:
s_cmp_gt_u32 s61, 7
s_cbranch_scc0 .L_k3_extra_wait_rank7_done
K3_TAIL_WAIT_SIGNAL 120
.L_k3_extra_wait_rank7_done:
.L_k3_extra_wait_done:
   s_mov_b64 exec, s[64:65]
   s_barrier
   s_waitcnt vmcnt(0) lgkmcnt(0)
   buffer_wbinvl1_vol
   s_waitcnt vmcnt(0)

s_lshl_b32 s77, s77, 4
s_mul_i32 s81, s80, 0x300
s_mul_i32 s82, s52, 0x300
v_add_u32 v250, s82, v[vgprSerial]
.L_k3_extra_reduce_loop:
v_cmp_lt_u32 vcc, v250, s76
s_and_saveexec_b64 s[84:85], vcc
s_cbranch_execz .L_k3_extra_reduce_done
v_mov_b32 v180, 0
v_mov_b32 v181, 0
v_mov_b32 v182, 0
v_mov_b32 v183, 0
v_mov_b32 v184, 0
v_mov_b32 v185, 0
v_mov_b32 v186, 0
v_mov_b32 v187, 0

K3_TAIL_ADDR_FROM_BASE s74, s75
K3_TAIL_LOAD_ACCUM_SIX
K3_TAIL_PACK_REDUCE_OUT
K3_TAIL_ADDR_FROM_BASE s72, s73
global_store_dwordx4 v[153:154], v[232:235], off
s_mov_b64 exec, s[84:85]
v_add_u32 v250, s81, v250
s_branch .L_k3_extra_reduce_loop
.L_k3_extra_reduce_done:
s_waitcnt vmcnt(0)
s_mov_b64 exec, s[84:85]
.L_k3_extra_reduce_endpgm:
s_endpgm

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s11, s24, 1
s_sub_u32 s8, s23, s20
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s8
/* Check if custom structure pointer is null */
s_cmp_eq_u64 s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0 // s[ExternalArgAddress] == 0 ?
s_cbranch_scc0 label_LoadExternalStruct            // branch if s[ExternalArgAddress] != 0

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s27, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s11, s11, 140
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s11
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dwordx16 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
s_load_dwordx4 s[36:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[40:41], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_load_dwordx4 s[92:95], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s11, s11, 128
s_add_u32 s[sgprExternalArgAddress], s[sgprExternalArgAddress], s11
s_addc_u32 s[sgprExternalArgAddress+1], s[sgprExternalArgAddress+1], 0x0
s_load_dwordx16 s[20:35], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x0
s_load_dwordx4 s[36:39], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x40
// Read Alpha Beta
s_load_dword s40, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x50
s_load_dword s41, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x60 
s_load_dwordx4 s[92:95], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x70
label_LoadExternalStructEnd:

s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
label_stop:
v_mov_b32 v6, MT0                                  // set MT0 into sgpr
v_mov_b32 v5, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v4, v6                               // v4 = ceil(v5 / v6)
v_rcp_iflag_f32 v4, v4                             // v4 = ceil(v5 / v6)
v_cvt_f32_u32 v7, v5                               // v4 = ceil(v5 / v6)
v_mul_f32 v4, v4, v7                               // v4 = ceil(v5 / v6)
v_cvt_u32_f32 v4, v4                               // v4 = ceil(v5 / v6)
v_mul_u32_u24 v7, v4, v6                           // v4 = ceil(v5 / v6)
v_sub_u32 v7, v5, v7                               // v4 = ceil(v5 / v6)
v_cmp_ne_u32 vcc, v7, 0                            // v4 = ceil(v5 / v6)
v_addc_co_u32 v4, vcc, v4, 0, vcc                  // ceil
v_mov_b32 v6, MT1                                  // set MT1 into sgpr
v_mov_b32 v5, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v4      // set back to numWorkGroup0
v_cvt_f32_u32 v4, v6                               // v4 = ceil(v5 / v6)
v_rcp_iflag_f32 v4, v4                             // v4 = ceil(v5 / v6)
v_cvt_f32_u32 v7, v5                               // v4 = ceil(v5 / v6)
v_mul_f32 v4, v4, v7                               // v4 = ceil(v5 / v6)
v_cvt_u32_f32 v4, v4                               // v4 = ceil(v5 / v6)
v_mul_u32_u24 v7, v4, v6                           // v4 = ceil(v5 / v6)
v_sub_u32 v7, v5, v7                               // v4 = ceil(v5 / v6)
v_cmp_ne_u32 vcc, v7, 0                            // v4 = ceil(v5 / v6)
v_addc_co_u32 v4, vcc, v4, 0, vcc                  // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v4      // set back to numWorkGroup1

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0x0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* Grouped Gemm: remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s56, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_mul_i32 s56, s56, s[sgprGSU]
v_cvt_f32_u32 v4, s56                              // s56 = s[sgprWorkGroup0] / s56
v_rcp_iflag_f32 v4, v4                             // s56 = s[sgprWorkGroup0] / s56
v_cvt_f32_u32 v5, s[sgprWorkGroup0]                // s56 = s[sgprWorkGroup0] / s56
v_mul_f32 v4, v4, v5                               // s56 = s[sgprWorkGroup0] / s56
v_cvt_u32_f32 v4, v4                               // s56 = s[sgprWorkGroup0] / s56
v_mul_u32_u24 v5, v4, s56                          // s56 = s[sgprWorkGroup0] / s56
v_sub_u32 v5, s[sgprWorkGroup0], v5                // s56 = s[sgprWorkGroup0] / s56
v_cmpx_eq_u32 exec, v5, s56                        // s56 = s[sgprWorkGroup0] / s56
v_add_u32 v4, 1, v4                                // s56 = s[sgprWorkGroup0] / s56
s_mov_b64 exec, -1                                 // s56 = s[sgprWorkGroup0] / s56
v_readfirstlane_b32 s56, v4
s_mov_b32 s[sgprWorkGroup2], s56
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s56, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s56, s56, s[sgprWorkGroup2]
s_mul_i32 s56, s56, s[sgprGSU]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s56
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v4, s[sgprNumWorkGroups0]            // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v4, v4                             // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v5, s[sgprWorkGroup0]                // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v4, v4, v5                               // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v4, v4                               // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v5, v4, s[sgprNumWorkGroups0]        // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v5, s[sgprWorkGroup0], v5                // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v5, s[sgprNumWorkGroups0]      // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v4, 1, v4                                // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // s56 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_readfirstlane_b32 s56, v4
s_mov_b32 s[sgprWorkGroup1], s56
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s56, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s56

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

/* K3 graph bucket: skip row tiles beyond K1 compact runtime active tile count.
 * Keep one dummy row tile alive so ranks with no local routes can still publish
 * their tail-reduce signal. */
s_cmp_eq_u64 s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0
s_cbranch_scc1 .L_k3_active_tile_gate_done
s_load_dword s86, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xc4
s_waitcnt lgkmcnt(0)
s_cmp_eq_u32 s86, 0
s_cbranch_scc1 .L_k3_active_tile_gate_done
s_load_dwordx2 s[90:91], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xc8
s_waitcnt lgkmcnt(0)
s_cmp_eq_u64 s[90:91], 0
s_cbranch_scc1 .L_k3_active_tile_gate_done
s_load_dword s88, s[90:91], 0x0
s_waitcnt lgkmcnt(0)
s_cmp_gt_u32 s88, 0
s_cbranch_scc1 .L_k3_active_tile_nonzero
s_mov_b32 s88, 1
.L_k3_active_tile_nonzero:
s_cmp_ge_u32 s[sgprWorkGroup1], s88
s_cbranch_scc0 .L_k3_active_tile_gate_done
s_endpgm
.L_k3_active_tile_gate_done:

s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                       // Alpha == 0 ?
s_cbranch_vccz label_AlphaNonZero                  // branch if alpha != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:

label_LocalReadAddr:

v_lshrrev_b32 v[vgprValuA_X0_I0], 6, v[vgprSerial]
v_readfirstlane_b32 s[sgprWaveiD], v[vgprValuA_X0_I0]
s_mov_b32 s[sgprMask], 0x10000

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */

v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64

/* local read addresses: final offsets a */
v_lshlrev_b32 v3, 4, v0
v_mov_b32 v[vgprLocalReadAddrA], v3
v_mov_b32 v243, v[vgprLocalReadAddrA]

/******************************************/
/* Begin setupNewTile, isPap=False           */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
; s_mov_b32 s60, 0x10000001L                         // magic number for WGM==8
; s_mul_hi_u32 s57, s[sgprWorkGroup1], s60           // s_magic mul
; s_mul_i32 s56, s[sgprWorkGroup1], s60              // s_magic mul
; s_lshr_b64 s[56:57], s[56:57], 31                  // sMagicDiv
; s_mul_i32 s57, s56, 8                              // quotient * non-magic divisor
; s_sub_u32 s57, s[sgprWorkGroup1], s57              // WorkGroup1=remainder
; s_mul_i32 s57, s57, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
; s_add_u32 s57, s57, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
; s_mul_hi_u32 s59, s[sgprNumWorkGroups1], s60       // s_magic mul
; s_mul_i32 s58, s[sgprNumWorkGroups1], s60          // s_magic mul
; s_lshr_b64 s[58:59], s[58:59], 31                  // sMagicDiv
; s_mul_i32 s59, 8, s58                              // quotient * non-magic divisor
; s_sub_u32 s60, s[sgprNumWorkGroups1], s59          // WorkGroup1=remainder
; s_cmp_eq_u32 s60, 0                                // remainder == 0 ?
; s_cmov_b32 s60, 8                                  // remainder = WGM if remainder == 0
; s_cmp_ge_u32 s56, s58                              // blockId >= numFullBlocks ?
; s_cselect_b32 s58, s60, 8
; v_cvt_f32_u32 v4, s58                              // s[sgprWorkGroup0] = s57 / s58
; v_rcp_iflag_f32 v4, v4                             // s[sgprWorkGroup0] = s57 / s58
; v_cvt_f32_u32 v5, s57                              // s[sgprWorkGroup0] = s57 / s58
; v_mul_f32 v4, v4, v5                               // s[sgprWorkGroup0] = s57 / s58
; v_cvt_u32_f32 v4, v4                               // s[sgprWorkGroup0] = s57 / s58
; v_mul_u32_u24 v5, v4, s58                          // s[sgprWorkGroup0] = s57 / s58
; v_sub_u32 v5, s57, v5                              // s[sgprWorkGroup0] = s57 / s58
; v_cmpx_eq_u32 exec, v5, s58                        // s[sgprWorkGroup0] = s57 / s58
; v_add_u32 v4, 1, v4                                // s[sgprWorkGroup0] = s57 / s58
; v_mov_b32 v5, 0                                    // s[sgprWorkGroup1] = s57 % s58
; s_mov_b64 exec, -1                                 // s[sgprWorkGroup0] = s57 / s58
; v_readfirstlane_b32 s[sgprWorkGroup0], v4
; v_readfirstlane_b32 s[sgprWorkGroup1], v5
; s_mul_i32 s56, s56, 8                              // blockId * WGM
; s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s56 // wg1 += blockId * WGM

/***********************读取m_indincs********************/
s_mul_hi_u32 s13, MT1, s[sgprWorkGroup1]
s_mul_i32    s12, MT1, s[sgprWorkGroup1]
s_lshl_b64 s[12:13], s[12:13], 2

s_add_u32 s[sgprSrdA+0], s98, s12
s_addc_u32 s[sgprSrdA+1], s99, s13
s_lshl_b32 s[sgprSrdA+2], s[sgprSizeJ], 2
s_sub_i32 s[sgprSrdA+2], s[sgprSrdA+2], s12
s_mov_b32 s[sgprSrdA+3], Srd127_96

v_mov_b32 v10, 0
buffer_load_dword v250, v10, s[sgprSrdA:sgprSrdA+3], 0, offen, offset:0x00
s_waitcnt vmcnt(0)
s_barrier

v_readfirstlane_b32 s[sgprScaleFlag], v250

/* global read addresses: tile offset assignment a */
v_and_b32 v6, 0xff, v[vgprSerial]
v_lshrrev_b32 v0, 7, v6
v_and_b32 v1, 0x7f, v6
v_lshlrev_b32 v1, 4, v1


/* global read addresses: tile offset assignment b */

v_and_b32 v4, 63, v[vgprSerial]          // v4 = v[vgprSerial] % 64
v_lshrrev_b32 v5, 6, v[vgprSerial]       // v5 = v[vgprSerial] / 64, 512 threads-> 8 wavefronts

v_lshrrev_b32 v3, 4, v4                  // v2 = (v[vgprSerial] % 64) / 16
v_and_b32 v2, 15, v4                     // v3 = (v[vgprSerial] % 64) % 16
/* gro-unroll *= glvw */
v_lshlrev_b32 v3, 0x4, v3                // v3 = v3 * 16

v_lshlrev_b32 v5, 5, v5                  // v5 = v5 * 32
v_add_co_u32 v2, vcc, v2, v5             // v2 = v2 + v5

/* global read addresses: tile offsets a */

v_mov_b32 v4, v0                                   // groA0I_0  0 - 31
_v_add_co_u32 v5, vcc, 2, v4                      // groA0I_1 += LSPA
_v_add_co_u32 v11, vcc, 2, v5                     // groA0I_1 += LSPA
_v_add_co_u32 v12, vcc, 2, v11                    // groA0I_1 += LSPA
_v_add_co_u32 v13, vcc, 2, v12                    // groA0I_1 += LSPA
_v_add_co_u32 v14, vcc, 2, v13                    // groA0I_1 += LSPA
_v_add_co_u32 v15, vcc, 2, v14                    // groA0I_1 += LSPA
_v_add_co_u32 v16, vcc, 2, v15                    // groA0I_1 += LSPA

/* global read addresses: tile offsets b */

v_mov_b32 v6, v2                                   // groB1J_0
_v_add_co_u32 v7, vcc, 16, v6                      // groB1J_1 += LSPB

/* global read addresses: unroll offsets a */

v_mov_b32 v8, v1                                   // groAL_0


/* global read addresses: unroll offsets b */

v_mov_b32 v9, v3                                   // groBL_0
_v_add_co_u32 v18, vcc, 64, v9                     // groBL_1 + LSCB


/* global read addresses: final offsets a */

v_lshrrev_b32 v[vgprPack5OffsetBaseA], 10, v8 // pack5 ko64
v_lshlrev_b32 v[vgprPack5OffsetBaseA], 18, v[vgprPack5OffsetBaseA] // ko64 * (4096 * 64)
v_and_b32 v[vgprPack5OffsetTmpA], 0x300, v8 // ks16 * 256
_v_add_u32 v[vgprPack5OffsetBaseA], v[vgprPack5OffsetBaseA], v[vgprPack5OffsetTmpA]
v_and_b32 v[vgprPack5OffsetTmpA], 15, v[vgprSerial] // logical ni for ASM lanes
v_lshrrev_b32 v10, 2, v[vgprPack5OffsetTmpA]
v_and_b32 v[vgprPack5OffsetTmpA], 3, v[vgprPack5OffsetTmpA]
v_lshlrev_b32 v[vgprPack5OffsetTmpA], 2, v[vgprPack5OffsetTmpA]
_v_add_u32 v[vgprPack5OffsetTmpA], v[vgprPack5OffsetTmpA], v10 // physical ni in shared transposed pack5
v_lshlrev_b32 v[vgprPack5OffsetTmpA], 4, v[vgprPack5OffsetTmpA]
_v_add_u32 v[vgprPack5OffsetBaseA], v[vgprPack5OffsetBaseA], v[vgprPack5OffsetTmpA]

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  8,  4, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+1,  8,  5, 10 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+2,  8,  11, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+3,  8,  12, 10 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+4,  8,  13, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+5,  8,  14, 10 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+6,  8,  15, 10 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+7,  8,  16, 10 // gROA_0_0_1_0


/* global read addresses: final offsets b */

s_mov_b32 s[sgprStructBitB], 0                     // init StructBitB to zero

s_cmp_ge_u32 s[sgprStrideB1J], 256                 // 
s_cmov_b32 s[sgprStructBitB], 8                    // 
s_cmp_ge_u32 s[sgprStrideB1J], 512                 // 
s_cmov_b32 s[sgprStructBitB], 9                    // 
s_cmp_ge_u32 s[sgprStrideB1J], 1024                // 
s_cmov_b32 s[sgprStructBitB], 10                   // 
s_cmp_ge_u32 s[sgprStrideB1J], 2048                // 
s_cmov_b32 s[sgprStructBitB], 11                   // 
s_cmp_ge_u32 s[sgprStrideB1J], 4096                // 
s_cmov_b32 s[sgprStructBitB], 12                   // 

s_lshl_b32 s[sgprStrideSwizzleB], 1, s[sgprStructBitB] // 

s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzleB]   // Ldb < StrideSwizzleB ?
s_cbranch_scc1 Skip_supply_offset1B                    // 
s_lshr_b32 s[sgprStrideStructB], s[sgprStrideB1J], s[sgprStructBitB]     // stride struct b
s_lshl_b32 s[sgprOffsetStructB], s[sgprStrideStructB], s[sgprStructBitB] // 
s_sub_u32 s[sgprOffsetStructB], s[sgprStrideB1J], s[sgprOffsetStructB]   // offset struct b
s_branch Skip_supply_offset2B                      // 
Skip_supply_offset1B:
s_mov_b32 s[sgprStrideStructB], 1                  // stride struct b
s_mov_b32 s[sgprOffsetStructB], 0                  // offset struct b
Skip_supply_offset2B:
GLOBAL_OFFSET_B vgprGlobalReadOffsetBForSWTL+0,  9,  6, 10 // gROB_0_0_0_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetBForSWTL+1,  18, 6, 10 // gROB_0_0_1_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetBForSWTL+2,  9,  7, 10 // gROB_0_0_0_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetBForSWTL+3,  18, 7, 10 // gROB_0_0_1_0

GLOBAL_STRUCT_INDEX_B  vgprGlobalReadOffsetB+0, 6
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+1, 9, 6
GLOBAL_STRUCT_INDEX_B  vgprGlobalReadOffsetB+2, 6
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+3, 18, 6
GLOBAL_STRUCT_INDEX_B  vgprGlobalReadOffsetB+4, 7
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+5, 9, 7
GLOBAL_STRUCT_INDEX_B  vgprGlobalReadOffsetB+6, 7
GLOBAL_STRUCT_OFFSET_B vgprGlobalReadOffsetB+7, 18, 7


/* global read addresses: addresses a */

/* V3 pack5 weight base: hidden tile + expert stride. */
s_mul_i32 s70, s[sgprWorkGroup0], 0x4000
s_mov_b32 s71, 0
s_mov_b32 s[sgprShadowLimitA+0], BufferLimit
s_mov_b32 s[sgprShadowLimitA+1], 0
s_mov_b32 s[sgprSrdA+2], BufferLimit

/***************token E_id * (N * K), 指向对应线程块所处理的token属于的E**************/
s_mul_hi_u32 s69, s[sgprStrideAK], s[sgprScaleFlag] // Stride*WG
s_mul_i32 s68, s[sgprStrideAK], s[sgprScaleFlag]  // Stride*WG
/************************************************************************************/

s_add_u32 s70, s70, s68                            // accum wg term to tilestart
s_addc_u32 s71, s71, s69                           // accum wg term to tilestart
                                                   // tileStart *= BPE (multiplier is 1, do nothing)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s70    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s71   // SRD base = Address+ tileStart1
//s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
//s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


s_mov_b64 s[sgprSrdA_forTailLoop+0:sgprSrdA_forTailLoop+1], s[sgprSrdA+0:sgprSrdA+1]
s_mov_b64 s[sgprSrdA_forTailLoop+2:sgprSrdA_forTailLoop+3], s[sgprSrdA+2:sgprSrdA+3]
s_mov_b64 s[sgprShadowLimitA_forTailLoop+0:sgprShadowLimitA_forTailLoop+1], s[sgprShadowLimitA+0:sgprShadowLimitA+1]

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 7 //  s[sgprLoopCounterL] = s[sgprSizesSum+0] / 128

;s_mov_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeA       // incrB (unrollIdx)    128
s_mov_b32 s[sgprGlobalReadIncsA+0], 0x80000
s_mul_i32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], s[sgprLoopCounterL]  

s_add_u32 s[sgprSrdA_forTailLoop+0], s[sgprSrdA_forTailLoop+0], s[sgprGlobalReadIncsA+0] // 
s_addc_u32 s[sgprSrdA_forTailLoop+1], s[sgprSrdA_forTailLoop+1], 0         // 
s_sub_u32 s[sgprShadowLimitA_forTailLoop+0], s[sgprShadowLimitA_forTailLoop+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA_forTailLoop+1], s[sgprShadowLimitA_forTailLoop+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA_forTailLoop+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA_forTailLoop+2], s[sgprShadowLimitA_forTailLoop+0]    // Move shadow to real if we are within 2^32


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s71, s[sgprWorkGroup1], 256           // WorkGroup[01] * MT
s_mul_i32 s70, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT
s_mul_hi_u32 s71, s70, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s70, s70, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride

s_mov_b32 s68, 1                                   // Init tensor size
s_mov_b32 s69, 0                                   // init tensor size
s_sub_u32 s67, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s66, constStrideBL, s67               // stride x (size-1)
s_mul_i32 s67, constStrideBL, s67                  // stride x (size-1)
s_add_u32 s68, s68, s67                            // sum tensor size
s_addc_u32 s69, s69, s66                           // sum tensor size
s_sub_u32 s67, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s66, s[sgprStrideB1J], s67            // stride x (size-1)
s_mul_i32 s67, s[sgprStrideB1J], s67               // stride x (size-1)
s_add_u32 s68, s68, s67                            // sum tensor size, s68 -> s[sgprTensor2dSizeB]
s_addc_u32 s69, s69, s66                           // sum tensor size, s69 -> s[sgprTensor2dSizeB+1]

s_sub_u32 s[sgprShadowLimitB_forTailLoop+0], s68, s70 // sub tileStart
s_subb_u32 s[sgprShadowLimitB_forTailLoop+1], s69, s71 // sub tileStart

//s_sub_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprTensor2dSizeB], s70 // sub tileStart
//s_subb_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprTensor2dSizeB+1], s71 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB_forTailLoop:sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop:sgprShadowLimitB_forTailLoop+1], 0x0 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprShadowLimitB_forTailLoop+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop+1], 0 // extend limit for pre-pad

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 7 //  s[sgprLoopCounterL] = s[sgprSizesSum+0] / 128
s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB      // 
s_mul_i32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s[sgprLoopCounterL] // 
s_sub_u32 s[sgprShadowLimitB_forTailLoop+0], s[sgprShadowLimitB_forTailLoop+0], s[sgprGlobalReadIncsB+0] // for Tail Loop
s_subb_u32 s[sgprShadowLimitB_forTailLoop+1], s[sgprShadowLimitB_forTailLoop+1], 0 // for Tail Loop

s_mul_hi_u32 s71, s[sgprWorkGroup1], 256           // WorkGroup[01] * MT
s_mul_i32 s70, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT
s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzleB] // ldb < StrideSwizzleB?
s_cbranch_scc1 Skip_fix_strideB1                   // 
s_lshr_b32 s[sgprStrideStructB], s[sgprStrideB1J], s[sgprStructBitB] // 
s_sub_u32 s[sgprShadowLimitB+0], s[sgprSizeJ], s70 // 
s_subb_u32 s[sgprShadowLimitB+1], 0, s71           // 
s_mul_hi_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+0], s[sgprStrideStructB] // 
s_mul_i32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprStrideStructB] // 
s_branch Skip_fix_strideB2                         // 
Skip_fix_strideB1:
s_lshr_b32 s[sgprStrideStructB], s[sgprStrideB1J], 0xd // 
s_sub_u32 s[sgprShadowLimitB+0], s[sgprSizeJ], s70 // 
s_subb_u32 s[sgprShadowLimitB+1], 0, s71           // 
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprStrideStructB] // 
Skip_fix_strideB2:
s_mul_hi_u32 s71, s70, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s70, s70, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

s_mul_hi_u32 s69, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s68, s[sgprStrideBK], s[sgprWorkGroup2]    // Stride*WG

s_add_u32 s70, s70, s68                              // accum wg term to tilestart
s_addc_u32 s71, s71, s69                             // accum wg term to tilestart
                                                     // tileStart *= BPE (multiplier is 1, do nothing)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s70      // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s71     // SRD base = Address+ tileStart1
//s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16           // pre-pad to make room for possible pointer shift
//s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0           // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdB+3], Srd127_96                   // Set bits 127_96 in SRD

s_add_u32 s[sgprStructBit1B], 16, s[sgprStructBitB]  // 
s_cmp_lt_u32 s[sgprStrideB1J], s[sgprStrideSwizzleB] // ldb < strideSwizzleB?
s_cbranch_scc1 skip_fix_setB                         // 
s_mov_b32 s[sgprStructNumB], 1                       // 
s_lshl_b32 s[sgprStructNumB], s[sgprStructNumB], s[sgprStructBit1B] // 
s_branch skip_fix_set2B                              // 
skip_fix_setB:
s_lshr_b32 s[sgprStrideStructB], s[sgprStrideB1J], 0xd // 
s_lshl_b32 s[sgprStructNumB], s[sgprStrideB1J], 0x0    // 
s_lshr_b32 s[sgprStructNumB], s[sgprStructNumB], s[sgprStrideStructB] // 
s_lshl_b32 s[sgprStructNumB], s[sgprStructNumB], 0x10  // 
skip_fix_set2B:
s_or_b32 s[sgprStructNumB], s[sgprStructNumB], 0x40000000 // 
s_or_b32 s[sgprSrdB+1], s[sgprSrdB+1], s[sgprStructNumB]  // 


/* global read addresses: increments a */
; s_mul_i32 s54, s[sgprGSU], DepthU*BpeAGR
s_mov_b32 s[sgprGlobalReadIncsA+0], 0x80000          // V3 pack5 incrA, 128 K step
; //s_mov_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeA    // incrA (unrollIdx)


/* global read addresses: increments b */
s_mul_i32 s54, s[sgprGSU], DepthU*BpeBGR
s_mov_b32 s[sgprGlobalReadIncsB+0], s54            // incrB (unrollIdx)
//s_mov_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeB    // incrB (unrollIdx)


/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

; v_and_b32 v4, 255, v[vgprSerial]  // v4 = v[vgprSerial] % 256
; v_and_b32 v1, 7, v4               // v1 = v4 % 8
; v_lshrrev_b32 v0, 3, v4           // v0 = v4 / 8
; v_lshlrev_b32 v1, 4, v1           // v1 = v1 * 16

; v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v0     // lwAL**(MTA + PAD)
; _v_add_lshl_u32 v[vgprLocalWriteAddrA], v1, v[vgprLocalWriteAddrA], 0x0 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe
; v_readfirstlane_b32 s[sgprLocalWriteAddrA], v[vgprLocalWriteAddrA]      // Copy lds write address VGPR to SGPR

; s_and_b32 s70, 3, s[sgprWaveiD]                    // s70 = s[sgprWaveiD] % 4   
; s_lshl_b32 s70, s70, 17                            // LdsWrap Offset: 8 dowrds -> 8 * 4bytes, 32个int8 
; s_or_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s70  // 

; s_mov_b32 s[sgprLocalWriteAddrA_ori], s[sgprLocalWriteAddrA]  // 
v_and_b32 v4, 0xff, v[vgprSerial]
v_lshlrev_b32 v1, 4, v4
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v1    
s_mov_b32 s[sgprLocalWriteAddrA_ori], s[sgprLocalWriteAddrA]

/* local write addresses: first offset b */

/* declare loop num iterations */

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 7  // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 128
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

/* prefetch: global -> local */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 ShadowInitStart_9                   // skip to ShadowInitStart iter b/c numIter==0

/* 生产者 warp */
s_cmp_le_i32 s[sgprWaveiD], 7                      //   <=7
s_cbranch_scc1 MmacWave                            // 
s_mov_b32 s[sgprLoopforPf], 0                      // 
Prefetch_Begin:                                    // 

s_mov_b32 m0, s[sgprLocalWriteAddrA]            // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 0x1000                            // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0,  lds // G -> Reg 0_0_1_0

/*加载指令发射完毕*/
s_barrier

s_add_u32 s[sgprLoopforPf], s[sgprLoopforPf], 1    // 

s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // 
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // 
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

s_add_u32 s[sgprLocalWriteAddrA], 0x8000, s[sgprLocalWriteAddrA] // 
s_and_b32 s90, s[sgprLocalWriteAddrA], 0x1ffff     // 
s_cmp_ge_u32 s90, s[sgprMask]                   // 
s_cmov_b32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA_ori] // 
s_cmp_lt_i32 s[sgprLoopforPf], 1                   // 
s_cbranch_scc1 SkipBarrier                         // 
s_waitcnt vmcnt(0)                                 // 
s_barrier                                          // 
SkipBarrier:                                       // 
s_cmp_lt_i32 s[sgprLoopforPf], s[sgprLoopCounterL] // 
s_cbranch_scc1 Prefetch_Begin                      // 
s_waitcnt vmcnt(0)                                 // 
s_barrier                                          // 
s_endpgm    


MmacWave:


/******************************************/
/* End setupNewTile, isPap=False             */
/******************************************/

ShadowInitStart_9: // 

s_cmp_le_i32 s[sgprWaveiD], 7                       
s_cbranch_scc1 MmacWave_Valid1                      
s_endpgm

MmacWave_Valid1:

s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0xfffffffe                // 
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0xfffffffe                // 
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s70, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s69, s70, s[sgprStrideC1J]            // CScale s70 by Stride
s_mul_i32 s68, s70, s[sgprStrideC1J]               // CScale s70 by Stride
s_lshl_b64 s[68:69], s[68:69], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s69       // add hi to SRD
s_mul_hi_u32 s69, s70, s[sgprStrideD1J]            // Scale s70 by Stride
s_mul_i32 s68, s70, s[sgprStrideD1J]               // Scale s70 by Stride
s_lshl_b64 s[68:69], s[68:69], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // add hi to SRD

s_mul_hi_u32 s69, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s68, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[68:69], s[68:69], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s69       // add hi to SRD
s_mul_hi_u32 s69, s[sgprWorkGroup2], s[sgprStrideDK] // Scale s[sgprWorkGroup2] by Stride
s_mul_i32 s68, s[sgprWorkGroup2], s[sgprStrideDK]  // Scale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[68:69], s[68:69], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // add hi to SRD



/* initC: remove C-tile 0-128 from pool */

/* initC: remove AB-tile 128-176 from pool */
v_mov_b32 v[vgprValuC+0], 0x0                      // initC
v_mov_b32 v[vgprValuC+1], 0x0                      // initC
v_mov_b32 v[vgprValuC+2], 0x0                      // initC
v_mov_b32 v[vgprValuC+3], 0x0                      // initC
v_mov_b32 v[vgprValuC+4], 0x0                      // initC
v_mov_b32 v[vgprValuC+5], 0x0                      // initC
v_mov_b32 v[vgprValuC+6], 0x0                      // initC
v_mov_b32 v[vgprValuC+7], 0x0                      // initC
v_mov_b32 v[vgprValuC+8], 0x0                      // initC
v_mov_b32 v[vgprValuC+9], 0x0                      // initC
v_mov_b32 v[vgprValuC+10], 0x0                     // initC
v_mov_b32 v[vgprValuC+11], 0x0                     // initC
v_mov_b32 v[vgprValuC+12], 0x0                     // initC
v_mov_b32 v[vgprValuC+13], 0x0                     // initC
v_mov_b32 v[vgprValuC+14], 0x0                     // initC
v_mov_b32 v[vgprValuC+15], 0x0                     // initC
v_mov_b32 v[vgprValuC+16], 0x0                     // initC
v_mov_b32 v[vgprValuC+17], 0x0                     // initC
v_mov_b32 v[vgprValuC+18], 0x0                     // initC
v_mov_b32 v[vgprValuC+19], 0x0                     // initC
v_mov_b32 v[vgprValuC+20], 0x0                     // initC
v_mov_b32 v[vgprValuC+21], 0x0                     // initC
v_mov_b32 v[vgprValuC+22], 0x0                     // initC
v_mov_b32 v[vgprValuC+23], 0x0                     // initC
v_mov_b32 v[vgprValuC+24], 0x0                     // initC
v_mov_b32 v[vgprValuC+25], 0x0                     // initC
v_mov_b32 v[vgprValuC+26], 0x0                     // initC
v_mov_b32 v[vgprValuC+27], 0x0                     // initC
v_mov_b32 v[vgprValuC+28], 0x0                     // initC
v_mov_b32 v[vgprValuC+29], 0x0                     // initC
v_mov_b32 v[vgprValuC+30], 0x0                     // initC
v_mov_b32 v[vgprValuC+31], 0x0                     // initC
v_mov_b32 v[vgprValuC+32], 0x0                     // initC
v_mov_b32 v[vgprValuC+33], 0x0                     // initC
v_mov_b32 v[vgprValuC+34], 0x0                     // initC
v_mov_b32 v[vgprValuC+35], 0x0                     // initC
v_mov_b32 v[vgprValuC+36], 0x0                     // initC
v_mov_b32 v[vgprValuC+37], 0x0                     // initC
v_mov_b32 v[vgprValuC+38], 0x0                     // initC
v_mov_b32 v[vgprValuC+39], 0x0                     // initC
v_mov_b32 v[vgprValuC+40], 0x0                     // initC
v_mov_b32 v[vgprValuC+41], 0x0                     // initC
v_mov_b32 v[vgprValuC+42], 0x0                     // initC
v_mov_b32 v[vgprValuC+43], 0x0                     // initC
v_mov_b32 v[vgprValuC+44], 0x0                     // initC
v_mov_b32 v[vgprValuC+45], 0x0                     // initC
v_mov_b32 v[vgprValuC+46], 0x0                     // initC
v_mov_b32 v[vgprValuC+47], 0x0                     // initC
v_mov_b32 v[vgprValuC+48], 0x0                     // initC
v_mov_b32 v[vgprValuC+49], 0x0                     // initC
v_mov_b32 v[vgprValuC+50], 0x0                     // initC
v_mov_b32 v[vgprValuC+51], 0x0                     // initC
v_mov_b32 v[vgprValuC+52], 0x0                     // initC
v_mov_b32 v[vgprValuC+53], 0x0                     // initC
v_mov_b32 v[vgprValuC+54], 0x0                     // initC
v_mov_b32 v[vgprValuC+55], 0x0                     // initC
v_mov_b32 v[vgprValuC+56], 0x0                     // initC
v_mov_b32 v[vgprValuC+57], 0x0                     // initC
v_mov_b32 v[vgprValuC+58], 0x0                     // initC
v_mov_b32 v[vgprValuC+59], 0x0                     // initC
v_mov_b32 v[vgprValuC+60], 0x0                     // initC
v_mov_b32 v[vgprValuC+61], 0x0                     // initC
v_mov_b32 v[vgprValuC+62], 0x0                     // initC
v_mov_b32 v[vgprValuC+63], 0x0                     // initC
v_mov_b32 v[vgprValuC+64], 0x0                     // initC
v_mov_b32 v[vgprValuC+65], 0x0                     // initC
v_mov_b32 v[vgprValuC+66], 0x0                     // initC
v_mov_b32 v[vgprValuC+67], 0x0                     // initC
v_mov_b32 v[vgprValuC+68], 0x0                     // initC
v_mov_b32 v[vgprValuC+69], 0x0                     // initC
v_mov_b32 v[vgprValuC+70], 0x0                     // initC
v_mov_b32 v[vgprValuC+71], 0x0                     // initC
v_mov_b32 v[vgprValuC+72], 0x0                     // initC
v_mov_b32 v[vgprValuC+73], 0x0                     // initC
v_mov_b32 v[vgprValuC+74], 0x0                     // initC
v_mov_b32 v[vgprValuC+75], 0x0                     // initC
v_mov_b32 v[vgprValuC+76], 0x0                     // initC
v_mov_b32 v[vgprValuC+77], 0x0                     // initC
v_mov_b32 v[vgprValuC+78], 0x0                     // initC
v_mov_b32 v[vgprValuC+79], 0x0                     // initC
v_mov_b32 v[vgprValuC+80], 0x0                     // initC
v_mov_b32 v[vgprValuC+81], 0x0                     // initC
v_mov_b32 v[vgprValuC+82], 0x0                     // initC
v_mov_b32 v[vgprValuC+83], 0x0                     // initC
v_mov_b32 v[vgprValuC+84], 0x0                     // initC
v_mov_b32 v[vgprValuC+85], 0x0                     // initC
v_mov_b32 v[vgprValuC+86], 0x0                     // initC
v_mov_b32 v[vgprValuC+87], 0x0                     // initC
v_mov_b32 v[vgprValuC+88], 0x0                     // initC
v_mov_b32 v[vgprValuC+89], 0x0                     // initC
v_mov_b32 v[vgprValuC+90], 0x0                     // initC
v_mov_b32 v[vgprValuC+91], 0x0                     // initC
v_mov_b32 v[vgprValuC+92], 0x0                     // initC
v_mov_b32 v[vgprValuC+93], 0x0                     // initC
v_mov_b32 v[vgprValuC+94], 0x0                     // initC
v_mov_b32 v[vgprValuC+95], 0x0                     // initC
v_mov_b32 v[vgprValuC+96], 0x0                     // initC
v_mov_b32 v[vgprValuC+97], 0x0                     // initC
v_mov_b32 v[vgprValuC+98], 0x0                     // initC
v_mov_b32 v[vgprValuC+99], 0x0                     // initC
v_mov_b32 v[vgprValuC+100], 0x0                    // initC
v_mov_b32 v[vgprValuC+101], 0x0                    // initC
v_mov_b32 v[vgprValuC+102], 0x0                    // initC
v_mov_b32 v[vgprValuC+103], 0x0                    // initC
v_mov_b32 v[vgprValuC+104], 0x0                    // initC
v_mov_b32 v[vgprValuC+105], 0x0                    // initC
v_mov_b32 v[vgprValuC+106], 0x0                    // initC
v_mov_b32 v[vgprValuC+107], 0x0                    // initC
v_mov_b32 v[vgprValuC+108], 0x0                    // initC
v_mov_b32 v[vgprValuC+109], 0x0                    // initC
v_mov_b32 v[vgprValuC+110], 0x0                    // initC
v_mov_b32 v[vgprValuC+111], 0x0                    // initC
v_mov_b32 v[vgprValuC+112], 0x0                    // initC
v_mov_b32 v[vgprValuC+113], 0x0                    // initC
v_mov_b32 v[vgprValuC+114], 0x0                    // initC
v_mov_b32 v[vgprValuC+115], 0x0                    // initC
v_mov_b32 v[vgprValuC+116], 0x0                    // initC
v_mov_b32 v[vgprValuC+117], 0x0                    // initC
v_mov_b32 v[vgprValuC+118], 0x0                    // initC
v_mov_b32 v[vgprValuC+119], 0x0                    // initC
v_mov_b32 v[vgprValuC+120], 0x0                    // initC
v_mov_b32 v[vgprValuC+121], 0x0                    // initC
v_mov_b32 v[vgprValuC+122], 0x0                    // initC
v_mov_b32 v[vgprValuC+123], 0x0                    // initC
v_mov_b32 v[vgprValuC+124], 0x0                    // initC
v_mov_b32 v[vgprValuC+125], 0x0                    // initC
v_mov_b32 v[vgprValuC+126], 0x0                    // initC
v_mov_b32 v[vgprValuC+127], 0x0                    // initC

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_10                   // Only branch on scc1
s_getpc_B64 s[68:69]                               // addr of next instr
s_add_u32 s70, PrefetchGlobalLastIterEnd_4, 0x4    // target branch offset
s_add_u32 s68, s68, s70                            // add target branch offset
s_addc_u32 s69, 0, s69                             // add high and carry
s_setpc_b64 s[68:69]                               // branch to PrefetchGlobalLastIterEnd_4
label_NoBranch_10:


buffer_load_dwordx4 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0

/*加载指令发射完毕， 通知生产者warp可以往后执行*/
s_barrier

s_waitcnt vmcnt(0)                                 

s_barrier //

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)


s_cmp_le_i32  s[sgprLoopCounterL], 1
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0


/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_11:

s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // do not enter LoopL

s_and_b32 s70, 7, s[sgprWaveiD]
s_cmp_ge_u32 s70, 4
s_cbranch_scc1 LoopBeginL_2

LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

label_0012: // LoopCopy1 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:4096 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:6144 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:8192 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:10240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:12288 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:14336 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:16384 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:20480 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:22528 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:24576 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:26624 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:28672 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:30720 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
MAC_32x4x2_X0_I0
s_barrier

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:5120 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:7168 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:11264 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:13312 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:15360 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:17408 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:19456 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:21504 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:23552 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:25600 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:27648 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:29696 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:31744 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_cmp_le_i32  s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0
s_waitcnt lgkmcnt(0)
MAC_32x4x2_X0_I1
s_barrier

s_waitcnt vmcnt(0)


/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/


/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_3                  // exit LoopL


/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

label_0013: // LoopCopy2 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:32768 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:34816 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:36864 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:38912 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:40960 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:43008 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:45056 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:47104 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:49152 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:51200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:53248 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:55296 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:57344 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:59392 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:61440 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:63488 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
MAC_32x4x2_X1_I0
s_barrier

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:33792 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:35840 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:37888 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:39936 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:41984 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:44032 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:46080 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:48128 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:50176 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:52224 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:54272 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:56320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:58368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:60416 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:62464 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:64512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_cmp_le_i32  s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0
s_waitcnt lgkmcnt(0)
MAC_32x4x2_X1_I1
s_barrier

s_waitcnt vmcnt(0)

/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
s_branch LoopEndL_2                                // exit unroll loopL (and skip oddexit)
/////////////////////////////////////////////////////
//////////////////////////////////////////////////////
LoopBeginL_2:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

label_00121: // LoopCopy1 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:4096 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:6144 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:8192 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:10240 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:12288 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:14336 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:16384 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:20480 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:22528 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:24576 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:26624 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:28672 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:30720 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
s_barrier
MAC_32x4x2_X0_I0

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:5120 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:7168 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:11264 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:13312 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:15360 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:17408 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:19456 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:21504 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:23552 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:25600 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:27648 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:29696 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:31744 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_cmp_le_i32  s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0
s_waitcnt lgkmcnt(0)
s_barrier
MAC_32x4x2_X0_I1

s_waitcnt vmcnt(0)


/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/


/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_cbranch_scc1 LoopEndL_oddexit_3                  // exit LoopL


/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

label_00131: // LoopCopy2 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:32768 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:34816 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:36864 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:38912 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:40960 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:43008 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:45056 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:47104 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:49152 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:51200 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:53248 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:55296 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:57344 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:59392 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:61440 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:63488 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
s_barrier

MAC_32x4x2_X1_I0

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:33792 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:35840 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:37888 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:39936 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:41984 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:44032 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:46080 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:48128 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
buffer_load_dwordx4 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0,idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0,idxen  offen offset:0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:50176 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:52224 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:54272 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:56320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:58368 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:60416 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:62464 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:64512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_cmp_le_i32  s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprSrdB+2], 0
s_cmov_b32 s[sgprGlobalReadIncsB+0], 0
s_waitcnt lgkmcnt(0)
s_barrier

MAC_32x4x2_X1_I1

s_waitcnt vmcnt(0)

/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_cbranch_scc0 LoopBeginL_2                        // restart LoopL


LoopEndL_oddexit_3: // unroll loop odditer exit

LoopEndL_2:

s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 OptNLL_End_14                       // Branch if Beta is not zero

s_cmp_eq_u32 s[sgprAlpha], 1.0                       // Alpha == 1.0 ?
s_cbranch_scc0 OptNLL_End_14                       // branch if alpha != 1

s_and_b32 s68, 255, s[sgprSizeI]                   // s68 = s[sgprSizeI] % 256
s_add_u32 s70, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s70                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rMT0
s_cmpk_gt_u32 s68, 0x0                             // rMT0 > 0
s_cbranch_scc1 OptNLL_End_14                       // jump if edges required

s_and_b32 s68, 255, s[sgprSizeJ]                   // s68 = s[sgprSizeJ] % 256
s_add_u32 s70, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s70                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rMT1
s_cmpk_gt_u32 s68, 0x0                             // rMT1 > 0
s_cbranch_scc1 OptNLL_End_14                       // jump if edges required

s_and_b32 s69, 127, s[sgprSizesSum+0]              // s69 = s[sgprSizesSum+0] % 128
s_cmp_eq_u32 s69, 0x0                              // numIterL == 0
s_cbranch_scc0 OptNLL_End_14                       // skip if tail loop required

s_barrier
; ValuC_INT32_TO_FP32

SMQUANT
; VgprValuC_ScaleA
; VgprValuC_ScaleB

FP32_TO_BF16_ALL 0

/* Stores for OptNLL */
Summation_End_OptNLL_15:
/* endSummation: add vgpr [128...198) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set StrideSwizzleB, UNDEF
.set StructNumB, UNDEF
.set StructBitB, UNDEF
.set StructBit1B, UNDEF
.set StrideStructB, UNDEF
.set OffsetStructB, UNDEF
.set ShadowLimitB_forTailLoop, UNDEF
/* computeStoreVgprs */
v_and_b32 v128, 63, v[vgprSerial]                  // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v129, 6, v[vgprSerial]               // v4 = v[vgprSerial] / 64
v_and_b32 v132, 15, v128                           // v5 = v0 % 16
v_lshrrev_b32 v133, 4, v128                        // v3 = v0 / 16
v_and_b32 v134, 0, v129                            // v134 = 0
s_mov_b32 s54, 128                                 //  
v_mul_lo_u32 v134, s54, v134                       // v134 = 0
v_lshlrev_b32 v128, 0, v132                        // v128 = (v[vgprSerial] % 64) % 16
_v_add_co_u32 v128, vcc, v128, v134                // row, v128 + 0
v_lshrrev_b32 v134, 0, v129                        // v129 = v[vgprSerial] / 64, waveId
s_mov_b32 s54, 32                                  //  
v_mul_lo_u32 v134, s54, v134                       // v134 = waveId * 32 
v_lshlrev_b32 v129, 0, v133                        // v129 = (v[vgprSerial] % 64) / 16
_v_add_co_u32 v129, vcc, v129, v134                // col, v129 + v134
/* K3COMBINE: keep D row_combine_ptrs before sgprAddressD is reused for stride offsets. */
s_mov_b64 s[92:93], s[sgprAddressD:sgprAddressD+1]
s_mov_b32 s94, BufferLimit
s_mov_b32 s95, Srd127_96
s_mul_i32 s[sgprStrideoffC+0], 8, s[sgprStridesC] //
s_mul_i32 s[sgprStrideoffC+1], 16, s[sgprStridesC] //
s_mul_i32 s[sgprStrideoffC+2], 24, s[sgprStridesC] //
s_mul_i32 s[sgprStrideoffD+0], 8, s[sgprStridesD] // 
s_mul_i32 s[sgprStrideoffD+1], 16, s[sgprStridesD] // 
s_mul_i32 s[sgprStrideoffD+2], 24, s[sgprStridesD] // 
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          // rowStart vgpr
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          // rowStart vgpr
v_mov_b32 v147, v129                               // K3COMBINE: local C-layout row unit

s_mul_i32 s54, 0x100, s[sgprWorkGroup0]            // s54 = wg0*MT0
_v_add_co_u32 v128, vcc, s54, v128                 // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s56, 0x100, s[sgprWorkGroup1]            // <- wg1*MT1
_v_add_co_u32 v129, vcc, s56, v129                 // coord1 = tid1*VW + wg1*MT1
GW_B0_E0_18:

/* K3COMBINE: half-tile LDS staging, then vectorized remote combine stores. */
s_lshl_b32 s54, s54, 1
v_lshlrev_b32 v200, 1, v132

s_cmp_le_i32 s[sgprWaveiD], 3
s_cbranch_scc0 .L_k3_stage_h0_skip
K3_STAGE_TILE_H0
.L_k3_stage_h0_skip:
s_waitcnt lgkmcnt(0)
s_barrier

s_cmp_le_i32 s[sgprWaveiD], 3
s_cbranch_scc0 .L_k3_store_h0_skip
v_mov_b32 v150, v[vgprSerial]
K3_STORE_STAGED_HALF 0, 512
.L_k3_store_h0_skip:
s_barrier

s_cmp_le_i32 s[sgprWaveiD], 3
s_cbranch_scc1 .L_k3_stage_h1_skip
v_mov_b32 v201, 128
K3_STAGE_TILE_H1
.L_k3_stage_h1_skip:
s_waitcnt lgkmcnt(0)
s_barrier

s_cmp_le_i32 s[sgprWaveiD], 3
s_cbranch_scc1 .L_k3_store_h1_skip
v_mov_b32 v201, 256
v_sub_u32 v150, v[vgprSerial], v201
K3_STORE_STAGED_HALF 1024, 512
.L_k3_store_h1_skip:
s_waitcnt vmcnt(0)
s_branch label_GW_End_20

/* K3COMBINE: direct combine epilogue. Keep row pointer loads outside the
 * column loop and advance four row addresses exactly like the original C
 * epilogue advances its four row offsets. */
.L_k3_direct_scalar_epilogue:
K3_LOAD_COMBINE_ADDR4 0, 4, 8, 12
K3_STORE4 v0, v1, v2, v3
K3_INC_ADDR4
K3_STORE4 v4, v5, v6, v7
K3_INC_ADDR4
K3_STORE4 v8, v9, v10, v11
K3_INC_ADDR4
K3_STORE4 v12, v13, v14, v15
K3_INC_ADDR4
K3_STORE4 v16, v17, v18, v19
K3_INC_ADDR4
K3_STORE4 v20, v21, v22, v23
K3_INC_ADDR4
K3_STORE4 v24, v25, v26, v27
K3_INC_ADDR4
K3_STORE4 v28, v29, v30, v31
K3_INC_ADDR4
K3_STORE4 v32, v33, v34, v35
K3_INC_ADDR4
K3_STORE4 v36, v37, v38, v39
K3_INC_ADDR4
K3_STORE4 v40, v41, v42, v43
K3_INC_ADDR4
K3_STORE4 v44, v45, v46, v47
K3_INC_ADDR4
K3_STORE4 v48, v49, v50, v51
K3_INC_ADDR4
K3_STORE4 v52, v53, v54, v55
K3_INC_ADDR4
K3_STORE4 v56, v57, v58, v59
K3_INC_ADDR4
K3_STORE4 v60, v61, v62, v63

K3_LOAD_COMBINE_ADDR4 16, 20, 24, 28
K3_STORE4 v64, v65, v66, v67
K3_INC_ADDR4
K3_STORE4 v68, v69, v70, v71
K3_INC_ADDR4
K3_STORE4 v72, v73, v74, v75
K3_INC_ADDR4
K3_STORE4 v76, v77, v78, v79
K3_INC_ADDR4
K3_STORE4 v80, v81, v82, v83
K3_INC_ADDR4
K3_STORE4 v84, v85, v86, v87
K3_INC_ADDR4
K3_STORE4 v88, v89, v90, v91
K3_INC_ADDR4
K3_STORE4 v92, v93, v94, v95
K3_INC_ADDR4
K3_STORE4 v96, v97, v98, v99
K3_INC_ADDR4
K3_STORE4 v100, v101, v102, v103
K3_INC_ADDR4
K3_STORE4 v104, v105, v106, v107
K3_INC_ADDR4
K3_STORE4 v108, v109, v110, v111
K3_INC_ADDR4
K3_STORE4 v112, v113, v114, v115
K3_INC_ADDR4
K3_STORE4 v116, v117, v118, v119
K3_INC_ADDR4
K3_STORE4 v120, v121, v122, v123
K3_INC_ADDR4
K3_STORE4 v124, v125, v126, v127
s_branch label_GW_End_20                           // jump to end
label_GW_End_20:
   s_waitcnt vmcnt(0) lgkmcnt(0)
   buffer_wbinvl1_vol
   s_waitcnt vmcnt(0)
   s_barrier

/* K3COMBINE tail reduce experiment. Every WG increments the per-rank done
 * counter after remote combine stores. Only the last local WG publishes this
 * rank's generation to peer signal slots and then uses its own 768 lanes to
 * reduce the local combine buffer into y. This avoids an in-kernel global wait
 * across all WGs, which would deadlock when the GEMM grid is larger than the
 * resident WG count. */
s_load_dwordx2 s[52:53], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x80
s_load_dwordx2 s[56:57], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x88
s_load_dwordx4 s[60:63], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0x90
s_load_dwordx2 s[72:73], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xa0
s_load_dwordx2 s[74:75], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xa8
s_load_dwordx4 s[76:79], s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xb0
s_load_dword s80, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xc0
s_waitcnt lgkmcnt(0)
K3_TAIL_APPLY_GRAPH_RUNTIME_STATE
s_cmp_eq_u32 s89, 0
s_cbranch_scc1 .L_k3_tail_main_done_counter_ready
s_and_b32 s87, s78, 15
s_lshl_b32 s87, s87, 3
s_add_u32 s52, s52, s87
s_addc_u32 s53, s53, 0
.L_k3_tail_main_done_counter_ready:
s_cmp_eq_u32 s62, 1
s_cbranch_scc0 .L_k3_tail_signal_done
s_cmp_eq_u64 s[52:53], 0
s_cbranch_scc1 .L_k3_tail_signal_done
s_cmp_eq_u64 s[56:57], 0
s_cbranch_scc1 .L_k3_tail_signal_done

v_cmp_eq_u32 vcc, v[vgprSerial], 0
s_and_saveexec_b64 s[64:65], vcc
s_cbranch_execz .L_k3_tail_lane0_done
v_mov_b32 v253, 0
ds_write_b32 v253, v253, offset:0
v_mov_b32 v238, 1
global_atomic_add v240, v253, v238, s[52:53] glc
s_waitcnt vmcnt(0)
v_add_u32 v240, 1, v240
v_cmp_eq_u32 vcc, v240, s60
s_and_saveexec_b64 s[68:69], vcc
s_cbranch_execz .L_k3_tail_not_last
s_waitcnt vmcnt(0) lgkmcnt(0)
buffer_wbinvl1_vol
s_waitcnt vmcnt(0)

s_cmp_gt_u32 s61, 0
s_cbranch_scc0 .L_k3_tail_rank0_done
K3_TAIL_ATOMIC_SIGNAL 0, s78
.L_k3_tail_rank0_done:
s_cmp_gt_u32 s61, 1
s_cbranch_scc0 .L_k3_tail_rank1_done
K3_TAIL_ATOMIC_SIGNAL 8, s78
.L_k3_tail_rank1_done:
s_cmp_gt_u32 s61, 2
s_cbranch_scc0 .L_k3_tail_rank2_done
K3_TAIL_ATOMIC_SIGNAL 16, s78
.L_k3_tail_rank2_done:
s_cmp_gt_u32 s61, 3
s_cbranch_scc0 .L_k3_tail_rank3_done
K3_TAIL_ATOMIC_SIGNAL 24, s78
.L_k3_tail_rank3_done:
s_cmp_gt_u32 s61, 4
s_cbranch_scc0 .L_k3_tail_rank4_done
K3_TAIL_ATOMIC_SIGNAL 32, s78
.L_k3_tail_rank4_done:
s_cmp_gt_u32 s61, 5
s_cbranch_scc0 .L_k3_tail_rank5_done
K3_TAIL_ATOMIC_SIGNAL 40, s78
.L_k3_tail_rank5_done:
s_cmp_gt_u32 s61, 6
s_cbranch_scc0 .L_k3_tail_rank6_done
K3_TAIL_ATOMIC_SIGNAL 48, s78
.L_k3_tail_rank6_done:
s_cmp_gt_u32 s61, 7
s_cbranch_scc0 .L_k3_tail_rank7_done
K3_TAIL_ATOMIC_SIGNAL 56, s78
.L_k3_tail_rank7_done:
s_waitcnt vmcnt(0)
s_cmp_gt_u32 s80, 0
s_cbranch_scc1 .L_k3_tail_signal_done

/* Debug/safety variant: avoid cross-wave LDS handoff. The wave that observed
 * the last-WG counter restores its full EXEC mask and performs the tail reduce
 * with 64 lanes. Other waves and non-last WGs exit below. */
s_mov_b64 exec, s[64:65]
s_cmp_eq_u32 s79, 1
s_cbranch_scc0 .L_k3_tail_signal_done
s_cmp_eq_u64 s[72:73], 0
s_cbranch_scc1 .L_k3_tail_signal_done
s_cmp_eq_u64 s[74:75], 0
s_cbranch_scc1 .L_k3_tail_signal_done

v_cmp_eq_u32 vcc, v[vgprSerial], 0
s_and_saveexec_b64 s[68:69], vcc
s_cbranch_execz .L_k3_tail_wait_done
s_cmp_gt_u32 s61, 0
s_cbranch_scc0 .L_k3_tail_wait_rank0_done
K3_TAIL_WAIT_SIGNAL 64
.L_k3_tail_wait_rank0_done:
s_cmp_gt_u32 s61, 1
s_cbranch_scc0 .L_k3_tail_wait_rank1_done
K3_TAIL_WAIT_SIGNAL 72
.L_k3_tail_wait_rank1_done:
s_cmp_gt_u32 s61, 2
s_cbranch_scc0 .L_k3_tail_wait_rank2_done
K3_TAIL_WAIT_SIGNAL 80
.L_k3_tail_wait_rank2_done:
s_cmp_gt_u32 s61, 3
s_cbranch_scc0 .L_k3_tail_wait_rank3_done
K3_TAIL_WAIT_SIGNAL 88
.L_k3_tail_wait_rank3_done:
s_cmp_gt_u32 s61, 4
s_cbranch_scc0 .L_k3_tail_wait_rank4_done
K3_TAIL_WAIT_SIGNAL 96
.L_k3_tail_wait_rank4_done:
s_cmp_gt_u32 s61, 5
s_cbranch_scc0 .L_k3_tail_wait_rank5_done
K3_TAIL_WAIT_SIGNAL 104
.L_k3_tail_wait_rank5_done:
s_cmp_gt_u32 s61, 6
s_cbranch_scc0 .L_k3_tail_wait_rank6_done
K3_TAIL_WAIT_SIGNAL 112
.L_k3_tail_wait_rank6_done:
s_cmp_gt_u32 s61, 7
s_cbranch_scc0 .L_k3_tail_wait_rank7_done
K3_TAIL_WAIT_SIGNAL 120
.L_k3_tail_wait_rank7_done:
.L_k3_tail_wait_done:
   s_mov_b64 exec, s[68:69]
   s_waitcnt vmcnt(0) lgkmcnt(0)
   buffer_wbinvl1_vol
   s_waitcnt vmcnt(0)

   s_lshl_b32 s77, s77, 4
v_mov_b32 v250, v[vgprSerial]
.L_k3_tail_reduce_loop:
v_cmp_lt_u32 vcc, v250, s76
s_and_saveexec_b64 s[84:85], vcc
s_cbranch_execz .L_k3_tail_reduce_done
v_mov_b32 v180, 0
v_mov_b32 v181, 0
v_mov_b32 v182, 0
v_mov_b32 v183, 0
v_mov_b32 v184, 0
v_mov_b32 v185, 0
v_mov_b32 v186, 0
v_mov_b32 v187, 0

K3_TAIL_ADDR_FROM_BASE s74, s75
K3_TAIL_LOAD_ACCUM_SIX
K3_TAIL_PACK_REDUCE_OUT
K3_TAIL_ADDR_FROM_BASE s72, s73
global_store_dwordx4 v[153:154], v[232:235], off
s_mov_b64 exec, s[84:85]
v_add_u32 v250, 64, v250
s_branch .L_k3_tail_reduce_loop
.L_k3_tail_reduce_done:
s_waitcnt vmcnt(0)
s_mov_b64 exec, s[84:85]
s_endpgm

.L_k3_tail_not_last:
s_mov_b64 exec, s[68:69]
.L_k3_tail_lane0_done:
s_mov_b64 exec, s[64:65]
.L_k3_tail_signal_done:
s_endpgm                                           // Kernel End


OptNLL_End_14:


/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

PrefetchGlobalLastIterEnd_4:

/******************************************/
/* Tail Loop                              */
/******************************************/

s_barrier
/* local write reset offsets a */

s_cmp_le_i32 s[sgprWaveiD], 7         ;  <=7             
s_cbranch_scc1 MmacWave_Valid2                      
s_endpgm

MmacWave_Valid2:

/* local write reset offsets b */

//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 127, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 128
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 SkipTailLoopL_7                     // skip to end of tail loop b/c numIter==0

s_cmp_eq_u32 s[sgprShadowLimitB_forTailLoop+1], 0  // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB_forTailLoop+0] // Move shadow to real if we are within 2^32

s_mov_b64 s[sgprSrdA+0:sgprSrdA+1], s[sgprSrdA_forTailLoop+0:sgprSrdA_forTailLoop+1]
s_mov_b64 s[sgprSrdA+2:sgprSrdA+3], s[sgprSrdA_forTailLoop+2:sgprSrdA_forTailLoop+3]

s_and_b32  s64, 7, s[sgprWaveiD]          // s64 = s[sgprWaveiD] % 8
s_lshr_b32 s64, s64, 2                    // s64 = s64 / 4
s_lshl_b32 s64, s64, 7                    // s64 = [(s[sgprWaveiD] % 8) / 4] * 128

s_mul_i32  s65, s64, 128                  // 
s_mul_i32  s64, s64, s[sgprStrideA0I]

v_add_co_u32 v[vgprGlobalReadOffsetA+0], vcc, s64, v[vgprGlobalReadOffsetA+0]
v_add_co_u32 v[vgprGlobalReadOffsetA+1], vcc, s64, v[vgprGlobalReadOffsetA+1]
v_add_co_u32 v[vgprGlobalReadOffsetA+2], vcc, s64, v[vgprGlobalReadOffsetA+2]
v_add_co_u32 v[vgprGlobalReadOffsetA+3], vcc, s64, v[vgprGlobalReadOffsetA+3]

v_add_co_u32 v[vgprLocalWriteAddrA], vcc, s65, v[vgprLocalWriteAddrA]

// 1. v[vgprG2LA+0] ~ v[vgprG2LA+3]
v_and_b32 v[vgprG2LA+0], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+0], 4, v[vgprG2LA+0]      // 
v_add_co_u32 v[vgprG2LA+0], vcc, 0, v[vgprG2LA+0]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+0], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+0], vcc, 0, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v[vgprG2LA+0], -0x1, v[vgprG2LA+0], s[64:65] // 
v_and_b32 v201, v[vgprSerial], 7                    // 
v_lshlrev_b32 v201, 4, v201                          // 
v_add_co_u32 v201, vcc, 1, v201                      // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]    // 
v_add_co_u32 v201, vcc, 1, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]             // 
v_and_b32 v202, v[vgprSerial], 7                    // 
v_lshlrev_b32 v202, 4, v202                          // 
v_add_co_u32 v202, vcc, 2, v202                      // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]    // 
v_add_co_u32 v202, vcc, 2, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]             // 
v_and_b32 v203, v[vgprSerial], 7                    // 
v_lshlrev_b32 v203, 4, v203                          // 
v_add_co_u32 v203, vcc, 3, v203                      // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]    // 
v_add_co_u32 v203, vcc, 3, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]             // 
v_and_b32 v[vgprG2LA+1], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+1], 4, v[vgprG2LA+1]      // 
v_add_co_u32 v[vgprG2LA+1], vcc, 4, v[vgprG2LA+1]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+1], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+1], vcc, 4, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v[vgprG2LA+1], -0x1, v[vgprG2LA+1], s[64:65] // 
v_and_b32 v204, v[vgprSerial], 7                    // 
v_lshlrev_b32 v204, 4, v204                          // 
v_add_co_u32 v204, vcc, 5, v204                      // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]    // 
v_add_co_u32 v204, vcc, 5, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]             // 
v_and_b32 v205, v[vgprSerial], 7                    // 
v_lshlrev_b32 v205, 4, v205                          // 
v_add_co_u32 v205, vcc, 6, v205                      // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]    // 
v_add_co_u32 v205, vcc, 6, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]             // 
v_and_b32 v206, v[vgprSerial], 7                    // 
v_lshlrev_b32 v206, 4, v206                          // 
v_add_co_u32 v206, vcc, 7, v206                      // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]    // 
v_add_co_u32 v206, vcc, 7, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]             // 
v_and_b32 v[vgprG2LA+2], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+2], 4, v[vgprG2LA+2]      // 
v_add_co_u32 v[vgprG2LA+2], vcc, 8, v[vgprG2LA+2]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+2], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+2], vcc, 8, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v[vgprG2LA+2], -0x1, v[vgprG2LA+2], s[64:65] // 
v_and_b32 v207, v[vgprSerial], 7                    // 
v_lshlrev_b32 v207, 4, v207                          // 
v_add_co_u32 v207, vcc, 9, v207                      // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]    // 
v_add_co_u32 v207, vcc, 9, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]             // 
v_and_b32 v208, v[vgprSerial], 7                    // 
v_lshlrev_b32 v208, 4, v208                          // 
v_add_co_u32 v208, vcc, 10, v208                     // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]    // 
v_add_co_u32 v208, vcc, 10, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]             // 
v_and_b32 v209, v[vgprSerial], 7                    // 
v_lshlrev_b32 v209, 4, v209                          // 
v_add_co_u32 v209, vcc, 11, v209                     // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]    // 
v_add_co_u32 v209, vcc, 11, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]             // 
v_and_b32 v[vgprG2LA+3], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+3], 4, v[vgprG2LA+3]      // 
v_add_co_u32 v[vgprG2LA+3], vcc, 12, v[vgprG2LA+3] // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+3], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+3], vcc, 12, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v[vgprG2LA+3], -0x1, v[vgprG2LA+3], s[64:65] // 
v_and_b32 v210, v[vgprSerial], 7                    // 
v_lshlrev_b32 v210, 4, v210                          // 
v_add_co_u32 v210, vcc, 13, v210                     // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]    // 
v_add_co_u32 v210, vcc, 13, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]             // 
v_and_b32 v211, v[vgprSerial], 7                    // 
v_lshlrev_b32 v211, 4, v211                          // 
v_add_co_u32 v211, vcc, 14, v211                     // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]    // 
v_add_co_u32 v211, vcc, 14, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]             // 
v_and_b32 v212, v[vgprSerial], 7                    // 
v_lshlrev_b32 v212, 4, v212                          // 
v_add_co_u32 v212, vcc, 15, v212                     // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]    // 
v_add_co_u32 v212, vcc, 15, v[vgprGlobalReadOffsetA+0] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]             // 

/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+0+0], v[vgprG2LA+0+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v201, v201, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v202, v202, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v203, v203, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+0+1], v[vgprG2LA+0+1], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v204, v204, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v205, v205, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v206, v206, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+0+2], v[vgprG2LA+0+2], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v207, v207, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v208, v208, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v209, v209, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+0+3], v[vgprG2LA+0+3], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v210, v210, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v211, v211, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v212, v212, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v201, 0x8, v201                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v201     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v202     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v203, 0x8, v203                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v203     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v204, 0x8, v204                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v204     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v205     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v206, 0x8, v206                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v206     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v207, 0x8, v207                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v207     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v208     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v209, 0x8, v209                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v209     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v210, 0x8, v210                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v210     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v211     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v212, 0x8, v212                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v212     // pack a sub 8-bit with dest


// 2. v[vgprG2LA+0+4] ~ v[vgprG2LA+3+4]
v_and_b32 v[vgprG2LA+0+4], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+0+4], 4, v[vgprG2LA+0+4]      // 
v_add_co_u32 v[vgprG2LA+0+4], vcc, 0, v[vgprG2LA+0+4]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+0+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+0+4], vcc, 0, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v[vgprG2LA+0+4], -0x1, v[vgprG2LA+0+4], s[64:65] // 
v_and_b32 v201, v[vgprSerial], 7                    // 
v_lshlrev_b32 v201, 4, v201                          // 
v_add_co_u32 v201, vcc, 1, v201                      // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]    // 
v_add_co_u32 v201, vcc, 1, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]             // 
v_and_b32 v202, v[vgprSerial], 7                    // 
v_lshlrev_b32 v202, 4, v202                          // 
v_add_co_u32 v202, vcc, 2, v202                      // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]    // 
v_add_co_u32 v202, vcc, 2, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]             // 
v_and_b32 v203, v[vgprSerial], 7                    // 
v_lshlrev_b32 v203, 4, v203                          // 
v_add_co_u32 v203, vcc, 3, v203                      // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]    // 
v_add_co_u32 v203, vcc, 3, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]             // 
v_and_b32 v[vgprG2LA+1+4], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+1+4], 4, v[vgprG2LA+1+4]      // 
v_add_co_u32 v[vgprG2LA+1+4], vcc, 4, v[vgprG2LA+1+4]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+1+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+1+4], vcc, 4, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v[vgprG2LA+1+4], -0x1, v[vgprG2LA+1+4], s[64:65] // 
v_and_b32 v204, v[vgprSerial], 7                    // 
v_lshlrev_b32 v204, 4, v204                          // 
v_add_co_u32 v204, vcc, 5, v204                      // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]    // 
v_add_co_u32 v204, vcc, 5, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]             // 
v_and_b32 v205, v[vgprSerial], 7                    // 
v_lshlrev_b32 v205, 4, v205                          // 
v_add_co_u32 v205, vcc, 6, v205                      // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]    // 
v_add_co_u32 v205, vcc, 6, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]             // 
v_and_b32 v206, v[vgprSerial], 7                    // 
v_lshlrev_b32 v206, 4, v206                          // 
v_add_co_u32 v206, vcc, 7, v206                      // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]    // 
v_add_co_u32 v206, vcc, 7, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]             // 
v_and_b32 v[vgprG2LA+2+4], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+2+4], 4, v[vgprG2LA+2+4]      // 
v_add_co_u32 v[vgprG2LA+2+4], vcc, 8, v[vgprG2LA+2+4]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+2+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+2+4], vcc, 8, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v[vgprG2LA+2+4], -0x1, v[vgprG2LA+2+4], s[64:65] // 
v_and_b32 v207, v[vgprSerial], 7                    // 
v_lshlrev_b32 v207, 4, v207                          // 
v_add_co_u32 v207, vcc, 9, v207                      // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]    // 
v_add_co_u32 v207, vcc, 9, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]             // 
v_and_b32 v208, v[vgprSerial], 7                    // 
v_lshlrev_b32 v208, 4, v208                          // 
v_add_co_u32 v208, vcc, 10, v208                     // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]    // 
v_add_co_u32 v208, vcc, 10, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]             // 
v_and_b32 v209, v[vgprSerial], 7                    // 
v_lshlrev_b32 v209, 4, v209                          // 
v_add_co_u32 v209, vcc, 11, v209                     // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]    // 
v_add_co_u32 v209, vcc, 11, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]             // 
v_and_b32 v[vgprG2LA+3+4], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+3+4], 4, v[vgprG2LA+3+4]      // 
v_add_co_u32 v[vgprG2LA+3+4], vcc, 12, v[vgprG2LA+3+4] // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+3+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+3+4], vcc, 12, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v[vgprG2LA+3+4], -0x1, v[vgprG2LA+3+4], s[64:65] // 
v_and_b32 v210, v[vgprSerial], 7                    // 
v_lshlrev_b32 v210, 4, v210                          // 
v_add_co_u32 v210, vcc, 13, v210                     // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]    // 
v_add_co_u32 v210, vcc, 13, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]             // 
v_and_b32 v211, v[vgprSerial], 7                    // 
v_lshlrev_b32 v211, 4, v211                          // 
v_add_co_u32 v211, vcc, 14, v211                     // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]    // 
v_add_co_u32 v211, vcc, 14, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]             // 
v_and_b32 v212, v[vgprSerial], 7                    // 
v_lshlrev_b32 v212, 4, v212                          // 
v_add_co_u32 v212, vcc, 15, v212                     // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]    // 
v_add_co_u32 v212, vcc, 15, v[vgprGlobalReadOffsetA+1] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]             // 

/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+0+0+4], v[vgprG2LA+0+0+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v201, v201, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v202, v202, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v203, v203, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+0+1+4], v[vgprG2LA+0+1+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v204, v204, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v205, v205, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v206, v206, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+0+2+4], v[vgprG2LA+0+2+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v207, v207, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v208, v208, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v209, v209, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+0+3+4], v[vgprG2LA+0+3+4], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v210, v210, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v211, v211, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v212, v212, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v201, 0x8, v201                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0+4], v[vgprG2LA+0+0+4], v201     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+0+0+4], v[vgprG2LA+0+0+4], v202     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v203, 0x8, v203                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0+4], v[vgprG2LA+0+0+4], v203     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v204, 0x8, v204                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1+4], v[vgprG2LA+0+1+4], v204     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+0+1+4], v[vgprG2LA+0+1+4], v205     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v206, 0x8, v206                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1+4], v[vgprG2LA+0+1+4], v206     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v207, 0x8, v207                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2+4], v[vgprG2LA+0+2+4], v207     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+0+2+4], v[vgprG2LA+0+2+4], v208     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v209, 0x8, v209                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2+4], v[vgprG2LA+0+2+4], v209     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v210, 0x8, v210                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3+4], v[vgprG2LA+0+3+4], v210     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+0+3+4], v[vgprG2LA+0+3+4], v211     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v212, 0x8, v212                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3+4], v[vgprG2LA+0+3+4], v212     // pack a sub 8-bit with dest


// 3. v[vgprG2LA+0+8] ~ v[vgprG2LA+3+8]
v_and_b32 v[vgprG2LA+0+8], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+0+8], 4, v[vgprG2LA+0+8]      // 
v_add_co_u32 v[vgprG2LA+0+8], vcc, 0, v[vgprG2LA+0+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+0+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+0+8], vcc, 0, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v[vgprG2LA+0+8], -0x1, v[vgprG2LA+0+8], s[64:65] // 
v_and_b32 v201, v[vgprSerial], 7                    // 
v_lshlrev_b32 v201, 4, v201                          // 
v_add_co_u32 v201, vcc, 1, v201                      // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]    // 
v_add_co_u32 v201, vcc, 1, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]             // 
v_and_b32 v202, v[vgprSerial], 7                    // 
v_lshlrev_b32 v202, 4, v202                          // 
v_add_co_u32 v202, vcc, 2, v202                      // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]    // 
v_add_co_u32 v202, vcc, 2, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]             // 
v_and_b32 v203, v[vgprSerial], 7                    // 
v_lshlrev_b32 v203, 4, v203                          // 
v_add_co_u32 v203, vcc, 3, v203                      // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]    // 
v_add_co_u32 v203, vcc, 3, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]             // 
v_and_b32 v[vgprG2LA+1+8], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+1+8], 4, v[vgprG2LA+1+8]      // 
v_add_co_u32 v[vgprG2LA+1+8], vcc, 4, v[vgprG2LA+1+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+1+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+1+8], vcc, 4, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v[vgprG2LA+1+8], -0x1, v[vgprG2LA+1+8], s[64:65] // 
v_and_b32 v204, v[vgprSerial], 7                    // 
v_lshlrev_b32 v204, 4, v204                          // 
v_add_co_u32 v204, vcc, 5, v204                      // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]    // 
v_add_co_u32 v204, vcc, 5, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]             // 
v_and_b32 v205, v[vgprSerial], 7                    // 
v_lshlrev_b32 v205, 4, v205                          // 
v_add_co_u32 v205, vcc, 6, v205                      // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]    // 
v_add_co_u32 v205, vcc, 6, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]             // 
v_and_b32 v206, v[vgprSerial], 7                    // 
v_lshlrev_b32 v206, 4, v206                          // 
v_add_co_u32 v206, vcc, 7, v206                      // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]    // 
v_add_co_u32 v206, vcc, 7, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]             // 
v_and_b32 v[vgprG2LA+2+8], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+2+8], 4, v[vgprG2LA+2+8]      // 
v_add_co_u32 v[vgprG2LA+2+8], vcc, 8, v[vgprG2LA+2+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+2+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+2+8], vcc, 8, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v[vgprG2LA+2+8], -0x1, v[vgprG2LA+2+8], s[64:65] // 
v_and_b32 v207, v[vgprSerial], 7                    // 
v_lshlrev_b32 v207, 4, v207                          // 
v_add_co_u32 v207, vcc, 9, v207                      // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]    // 
v_add_co_u32 v207, vcc, 9, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]             // 
v_and_b32 v208, v[vgprSerial], 7                    // 
v_lshlrev_b32 v208, 4, v208                          // 
v_add_co_u32 v208, vcc, 10, v208                     // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]    // 
v_add_co_u32 v208, vcc, 10, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]             // 
v_and_b32 v209, v[vgprSerial], 7                    // 
v_lshlrev_b32 v209, 4, v209                          // 
v_add_co_u32 v209, vcc, 11, v209                     // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]    // 
v_add_co_u32 v209, vcc, 11, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]             // 
v_and_b32 v[vgprG2LA+3+8], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+3+8], 4, v[vgprG2LA+3+8]      // 
v_add_co_u32 v[vgprG2LA+3+8], vcc, 12, v[vgprG2LA+3+8] // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+3+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+3+8], vcc, 12, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v[vgprG2LA+3+8], -0x1, v[vgprG2LA+3+8], s[64:65] // 
v_and_b32 v210, v[vgprSerial], 7                    // 
v_lshlrev_b32 v210, 4, v210                          // 
v_add_co_u32 v210, vcc, 13, v210                     // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]    // 
v_add_co_u32 v210, vcc, 13, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]             // 
v_and_b32 v211, v[vgprSerial], 7                    // 
v_lshlrev_b32 v211, 4, v211                          // 
v_add_co_u32 v211, vcc, 14, v211                     // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]    // 
v_add_co_u32 v211, vcc, 14, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]             // 
v_and_b32 v212, v[vgprSerial], 7                    // 
v_lshlrev_b32 v212, 4, v212                          // 
v_add_co_u32 v212, vcc, 15, v212                     // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]    // 
v_add_co_u32 v212, vcc, 15, v[vgprGlobalReadOffsetA+2] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]             // 

/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+0+0+8], v[vgprG2LA+0+0+8], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v201, v201, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v202, v202, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v203, v203, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+0+1+8], v[vgprG2LA+0+1+8], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v204, v204, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v205, v205, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v206, v206, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+0+2+8], v[vgprG2LA+0+2+8], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v207, v207, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v208, v208, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v209, v209, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+0+3+8], v[vgprG2LA+0+3+8], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v210, v210, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v211, v211, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v212, v212, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v201, 0x8, v201                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0+8], v[vgprG2LA+0+0+8], v201     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+0+0+8], v[vgprG2LA+0+0+8], v202     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v203, 0x8, v203                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0+8], v[vgprG2LA+0+0+8], v203     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v204, 0x8, v204                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1+8], v[vgprG2LA+0+1+8], v204     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+0+1+8], v[vgprG2LA+0+1+8], v205     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v206, 0x8, v206                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1+8], v[vgprG2LA+0+1+8], v206     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v207, 0x8, v207                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2+8], v[vgprG2LA+0+2+8], v207     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+0+2+8], v[vgprG2LA+0+2+8], v208     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v209, 0x8, v209                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2+8], v[vgprG2LA+0+2+8], v209     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v210, 0x8, v210                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3+8], v[vgprG2LA+0+3+8], v210     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+0+3+8], v[vgprG2LA+0+3+8], v211     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v212, 0x8, v212                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3+8], v[vgprG2LA+0+3+8], v212     // pack a sub 8-bit with dest


// 4. v[vgprG2LA+0+12] ~ v[vgprG2LA+3+12]
v_and_b32 v[vgprG2LA+0+12], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+0+12], 4, v[vgprG2LA+0+12]      // 
v_add_co_u32 v[vgprG2LA+0+12], vcc, 0, v[vgprG2LA+0+12]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+0+12], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+0+12], vcc, 0, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v[vgprG2LA+0+12], -0x1, v[vgprG2LA+0+12], s[64:65] // 
v_and_b32 v201, v[vgprSerial], 7                    // 
v_lshlrev_b32 v201, 4, v201                          // 
v_add_co_u32 v201, vcc, 1, v201                      // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]    // 
v_add_co_u32 v201, vcc, 1, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]             // 
v_and_b32 v202, v[vgprSerial], 7                    // 
v_lshlrev_b32 v202, 4, v202                          // 
v_add_co_u32 v202, vcc, 2, v202                      // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]    // 
v_add_co_u32 v202, vcc, 2, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]             // 
v_and_b32 v203, v[vgprSerial], 7                    // 
v_lshlrev_b32 v203, 4, v203                          // 
v_add_co_u32 v203, vcc, 3, v203                      // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]    // 
v_add_co_u32 v203, vcc, 3, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]             // 
v_and_b32 v[vgprG2LA+1+12], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+1+12], 4, v[vgprG2LA+1+12]      // 
v_add_co_u32 v[vgprG2LA+1+12], vcc, 4, v[vgprG2LA+1+12]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+1+12], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+1+12], vcc, 4, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v[vgprG2LA+1+12], -0x1, v[vgprG2LA+1+12], s[64:65] // 
v_and_b32 v204, v[vgprSerial], 7                    // 
v_lshlrev_b32 v204, 4, v204                          // 
v_add_co_u32 v204, vcc, 5, v204                      // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]    // 
v_add_co_u32 v204, vcc, 5, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]             // 
v_and_b32 v205, v[vgprSerial], 7                    // 
v_lshlrev_b32 v205, 4, v205                          // 
v_add_co_u32 v205, vcc, 6, v205                      // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]    // 
v_add_co_u32 v205, vcc, 6, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]             // 
v_and_b32 v206, v[vgprSerial], 7                    // 
v_lshlrev_b32 v206, 4, v206                          // 
v_add_co_u32 v206, vcc, 7, v206                      // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]    // 
v_add_co_u32 v206, vcc, 7, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]             // 
v_and_b32 v[vgprG2LA+2+12], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+2+12], 4, v[vgprG2LA+2+12]      // 
v_add_co_u32 v[vgprG2LA+2+12], vcc, 8, v[vgprG2LA+2+12]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+2+12], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+2+12], vcc, 8, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v[vgprG2LA+2+12], -0x1, v[vgprG2LA+2+12], s[64:65] // 
v_and_b32 v207, v[vgprSerial], 7                    // 
v_lshlrev_b32 v207, 4, v207                          // 
v_add_co_u32 v207, vcc, 9, v207                      // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]    // 
v_add_co_u32 v207, vcc, 9, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]             // 
v_and_b32 v208, v[vgprSerial], 7                    // 
v_lshlrev_b32 v208, 4, v208                          // 
v_add_co_u32 v208, vcc, 10, v208                     // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]    // 
v_add_co_u32 v208, vcc, 10, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]             // 
v_and_b32 v209, v[vgprSerial], 7                    // 
v_lshlrev_b32 v209, 4, v209                          // 
v_add_co_u32 v209, vcc, 11, v209                     // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]    // 
v_add_co_u32 v209, vcc, 11, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]             // 
v_and_b32 v[vgprG2LA+3+12], v[vgprSerial], 7          // 
v_lshlrev_b32 v[vgprG2LA+3+12], 4, v[vgprG2LA+3+12]      // 
v_add_co_u32 v[vgprG2LA+3+12], vcc, 12, v[vgprG2LA+3+12] // 
v_cmp_lt_u32 s[64:65], v[vgprG2LA+3+12], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LA+3+12], vcc, 12, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v[vgprG2LA+3+12], -0x1, v[vgprG2LA+3+12], s[64:65] // 
v_and_b32 v210, v[vgprSerial], 7                    // 
v_lshlrev_b32 v210, 4, v210                          // 
v_add_co_u32 v210, vcc, 13, v210                     // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]    // 
v_add_co_u32 v210, vcc, 13, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]             // 
v_and_b32 v211, v[vgprSerial], 7                    // 
v_lshlrev_b32 v211, 4, v211                          // 
v_add_co_u32 v211, vcc, 14, v211                     // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]    // 
v_add_co_u32 v211, vcc, 14, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]             // 
v_and_b32 v212, v[vgprSerial], 7                    // 
v_lshlrev_b32 v212, 4, v212                          // 
v_add_co_u32 v212, vcc, 15, v212                     // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]    // 
v_add_co_u32 v212, vcc, 15, v[vgprGlobalReadOffsetA+3] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]             // 

/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+0+0+12], v[vgprG2LA+0+0+12], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v201, v201, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v202, v202, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v203, v203, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+0+1+12], v[vgprG2LA+0+1+12], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v204, v204, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v205, v205, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v206, v206, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+0+2+12], v[vgprG2LA+0+2+12], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v207, v207, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v208, v208, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v209, v209, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+0+3+12], v[vgprG2LA+0+3+12], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v210, v210, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v211, v211, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v212, v212, s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v201, 0x8, v201                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0+12], v[vgprG2LA+0+0+12], v201     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+0+0+12], v[vgprG2LA+0+0+12], v202     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v203, 0x8, v203                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0+12], v[vgprG2LA+0+0+12], v203     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v204, 0x8, v204                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1+12], v[vgprG2LA+0+1+12], v204     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+0+1+12], v[vgprG2LA+0+1+12], v205     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v206, 0x8, v206                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1+12], v[vgprG2LA+0+1+12], v206     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v207, 0x8, v207                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2+12], v[vgprG2LA+0+2+12], v207     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+0+2+12], v[vgprG2LA+0+2+12], v208     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v209, 0x8, v209                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2+12], v[vgprG2LA+0+2+12], v209     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v210, 0x8, v210                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3+12], v[vgprG2LA+0+3+12], v210     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+0+3+12], v[vgprG2LA+0+3+12], v211     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v212, 0x8, v212                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3+12], v[vgprG2LA+0+3+12], v212     // pack a sub 8-bit with dest



/* local write a */

ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 8192
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8192 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12288 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 8192

s_waitcnt lgkmcnt(0)
s_barrier //


v_mov_b32 v[vgprGlobalReadOffsetB+0], v[vgprGlobalReadOffsetBForSWTL+0]
v_mov_b32 v[vgprGlobalReadOffsetB+1], v[vgprGlobalReadOffsetBForSWTL+1]
v_mov_b32 v[vgprGlobalReadOffsetB+2], v[vgprGlobalReadOffsetBForSWTL+2]
v_mov_b32 v[vgprGlobalReadOffsetB+3], v[vgprGlobalReadOffsetBForSWTL+3]

v_and_b32 v217, 63, v[vgprSerial]                  // v217 = v[vgprSerial] % 64
v_lshrrev_b32 v218, 6, v[vgprSerial]               // v218 = v[vgprSerial] / 64

v_lshrrev_b32 v219, 4, v217                        // v219 = v217 / 16
v_and_b32 v220, 15, v217                           // v220 = v217 % 16

// 1. v[vgprG2LB+0] ~ v[vgprG2LB+3]
v_mov_b32 v[vgprG2LB+0], v219          // 
v_lshlrev_b32 v[vgprG2LB+0], 4, v[vgprG2LB+0]      // 
v_add_co_u32 v[vgprG2LB+0], vcc, 0, v[vgprG2LB+0]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+0], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+0], vcc, 0, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+0], -0x1, v[vgprG2LB+0], s[64:65] // 
v_mov_b32 v201, v219                    // 
v_lshlrev_b32 v201, 4, v201                          // 
v_add_co_u32 v201, vcc, 1, v201                      // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]    // 
v_add_co_u32 v201, vcc, 1, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]             // 
v_mov_b32 v202, v219                    // 
v_lshlrev_b32 v202, 4, v202                          // 
v_add_co_u32 v202, vcc, 2, v202                      // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]    // 
v_add_co_u32 v202, vcc, 2, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]             // 
v_mov_b32 v203, v219                    // 
v_lshlrev_b32 v203, 4, v203                          // 
v_add_co_u32 v203, vcc, 3, v203                      // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]    // 
v_add_co_u32 v203, vcc, 3, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]             // 
v_mov_b32 v[vgprG2LB+1], v219          // 
v_lshlrev_b32 v[vgprG2LB+1], 4, v[vgprG2LB+1]      // 
v_add_co_u32 v[vgprG2LB+1], vcc, 4, v[vgprG2LB+1]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+1], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+1], vcc, 4, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+1], -0x1, v[vgprG2LB+1], s[64:65] // 
v_mov_b32 v204, v219                    // 
v_lshlrev_b32 v204, 4, v204                          // 
v_add_co_u32 v204, vcc, 5, v204                      // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]    // 
v_add_co_u32 v204, vcc, 5, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]             // 
v_mov_b32 v205, v219                    // 
v_lshlrev_b32 v205, 4, v205                          // 
v_add_co_u32 v205, vcc, 6, v205                      // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]    // 
v_add_co_u32 v205, vcc, 6, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]             // 
v_mov_b32 v206, v219                    // 
v_lshlrev_b32 v206, 4, v206                          // 
v_add_co_u32 v206, vcc, 7, v206                      // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]    // 
v_add_co_u32 v206, vcc, 7, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]             // 
v_mov_b32 v[vgprG2LB+2], v219          // 
v_lshlrev_b32 v[vgprG2LB+2], 4, v[vgprG2LB+2]      // 
v_add_co_u32 v[vgprG2LB+2], vcc, 8, v[vgprG2LB+2]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+2], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+2], vcc, 8, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+2], -0x1, v[vgprG2LB+2], s[64:65] // 
v_mov_b32 v207, v219                    // 
v_lshlrev_b32 v207, 4, v207                          // 
v_add_co_u32 v207, vcc, 9, v207                      // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]    // 
v_add_co_u32 v207, vcc, 9, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]             // 
v_mov_b32 v208, v219                    // 
v_lshlrev_b32 v208, 4, v208                          // 
v_add_co_u32 v208, vcc, 10, v208                     // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]    // 
v_add_co_u32 v208, vcc, 10, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]             // 
v_mov_b32 v209, v219                    // 
v_lshlrev_b32 v209, 4, v209                          // 
v_add_co_u32 v209, vcc, 11, v209                     // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]    // 
v_add_co_u32 v209, vcc, 11, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]             // 
v_mov_b32 v[vgprG2LB+3], v219          // 
v_lshlrev_b32 v[vgprG2LB+3], 4, v[vgprG2LB+3]      // 
v_add_co_u32 v[vgprG2LB+3], vcc, 12, v[vgprG2LB+3] // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+3], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+3], vcc, 12, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v[vgprG2LB+3], -0x1, v[vgprG2LB+3], s[64:65] // 
v_mov_b32 v210, v219                    // 
v_lshlrev_b32 v210, 4, v210                          // 
v_add_co_u32 v210, vcc, 13, v210                     // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]    // 
v_add_co_u32 v210, vcc, 13, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]             // 
v_mov_b32 v211, v219                    // 
v_lshlrev_b32 v211, 4, v211                          // 
v_add_co_u32 v211, vcc, 14, v211                     // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]    // 
v_add_co_u32 v211, vcc, 14, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]             // 
v_mov_b32 v212, v219                    // 
v_lshlrev_b32 v212, 4, v212                          // 
v_add_co_u32 v212, vcc, 15, v212                     // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]    // 
v_add_co_u32 v212, vcc, 15, v[vgprGlobalReadOffsetB+0] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]             // 

/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+0+0], v[vgprG2LB+0+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v201, v201, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v202, v202, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v203, v203, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+0+1], v[vgprG2LB+0+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v204, v204, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v205, v205, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v206, v206, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+0+2], v[vgprG2LB+0+2], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v207, v207, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v208, v208, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v209, v209, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+0+3], v[vgprG2LB+0+3], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v210, v210, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v211, v211, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v212, v212, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v201, 0x8, v201                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v201     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v202     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v203, 0x8, v203                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v203     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v204, 0x8, v204                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v204     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v205     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v206, 0x8, v206                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v206     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v207, 0x8, v207                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v207     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v208     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v209, 0x8, v209                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v209     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v210, 0x8, v210                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v210     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v211     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v212, 0x8, v212                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v212     // pack a sub 8-bit with dest


// 2. v[vgprG2LB+0+4] ~ v[vgprG2LB+3+4]
v_mov_b32 v[vgprG2LB+0+4], v219          // 
v_lshlrev_b32 v[vgprG2LB+0+4], 4, v[vgprG2LB+0+4]      // 
v_add_co_u32 v[vgprG2LB+0+4], vcc, 64, v[vgprG2LB+0+4]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+0+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+0+4], vcc, 0, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v[vgprG2LB+0+4], -0x1, v[vgprG2LB+0+4], s[64:65] // 
v_mov_b32 v201, v219                    // 
v_lshlrev_b32 v201, 4, v201                          // 
v_add_co_u32 v201, vcc, 65, v201                      // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]    // 
v_add_co_u32 v201, vcc, 1, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]             // 
v_mov_b32 v202, v219                    // 
v_lshlrev_b32 v202, 4, v202                          // 
v_add_co_u32 v202, vcc, 66, v202                      // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]    // 
v_add_co_u32 v202, vcc, 2, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]             // 
v_mov_b32 v203, v219                    // 
v_lshlrev_b32 v203, 4, v203                          // 
v_add_co_u32 v203, vcc, 67, v203                      // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]    // 
v_add_co_u32 v203, vcc, 3, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]             // 
v_mov_b32 v[vgprG2LB+1+4], v219          // 
v_lshlrev_b32 v[vgprG2LB+1+4], 4, v[vgprG2LB+1+4]      // 
v_add_co_u32 v[vgprG2LB+1+4], vcc, 68, v[vgprG2LB+1+4]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+1+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+1+4], vcc, 4, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v[vgprG2LB+1+4], -0x1, v[vgprG2LB+1+4], s[64:65] // 
v_mov_b32 v204, v219                    // 
v_lshlrev_b32 v204, 4, v204                          // 
v_add_co_u32 v204, vcc, 69, v204                      // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]    // 
v_add_co_u32 v204, vcc, 5, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]             // 
v_mov_b32 v205, v219                    // 
v_lshlrev_b32 v205, 4, v205                          // 
v_add_co_u32 v205, vcc, 70, v205                      // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]    // 
v_add_co_u32 v205, vcc, 6, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]             // 
v_mov_b32 v206, v219                    // 
v_lshlrev_b32 v206, 4, v206                          // 
v_add_co_u32 v206, vcc, 71, v206                      // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]    // 
v_add_co_u32 v206, vcc, 7, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]             // 
v_mov_b32 v[vgprG2LB+2+4], v219          // 
v_lshlrev_b32 v[vgprG2LB+2+4], 4, v[vgprG2LB+2+4]      // 
v_add_co_u32 v[vgprG2LB+2+4], vcc, 72, v[vgprG2LB+2+4]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+2+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+2+4], vcc, 8, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v[vgprG2LB+2+4], -0x1, v[vgprG2LB+2+4], s[64:65] // 
v_mov_b32 v207, v219                    // 
v_lshlrev_b32 v207, 4, v207                          // 
v_add_co_u32 v207, vcc, 73, v207                      // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]    // 
v_add_co_u32 v207, vcc, 9, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]             // 
v_mov_b32 v208, v219                    // 
v_lshlrev_b32 v208, 4, v208                          // 
v_add_co_u32 v208, vcc, 74, v208                     // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]    // 
v_add_co_u32 v208, vcc, 10, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]             // 
v_mov_b32 v209, v219                    // 
v_lshlrev_b32 v209, 4, v209                          // 
v_add_co_u32 v209, vcc, 75, v209                     // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]    // 
v_add_co_u32 v209, vcc, 11, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]             // 
v_mov_b32 v[vgprG2LB+3+4], v219          // 
v_lshlrev_b32 v[vgprG2LB+3+4], 4, v[vgprG2LB+3+4]      // 
v_add_co_u32 v[vgprG2LB+3+4], vcc, 76, v[vgprG2LB+3+4] // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+3+4], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+3+4], vcc, 12, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v[vgprG2LB+3+4], -0x1, v[vgprG2LB+3+4], s[64:65] // 
v_mov_b32 v210, v219                    // 
v_lshlrev_b32 v210, 4, v210                          // 
v_add_co_u32 v210, vcc, 77, v210                     // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]    // 
v_add_co_u32 v210, vcc, 13, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]             // 
v_mov_b32 v211, v219                    // 
v_lshlrev_b32 v211, 4, v211                          // 
v_add_co_u32 v211, vcc, 78, v211                     // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]    // 
v_add_co_u32 v211, vcc, 14, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]             // 
v_mov_b32 v212, v219                    // 
v_lshlrev_b32 v212, 4, v212                          // 
v_add_co_u32 v212, vcc, 79, v212                     // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]    // 
v_add_co_u32 v212, vcc, 15, v[vgprGlobalReadOffsetB+1] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]             // 

/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+0+0+4], v[vgprG2LB+0+0+4], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v201, v201, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v202, v202, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v203, v203, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+0+1+4], v[vgprG2LB+0+1+4], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v204, v204, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v205, v205, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v206, v206, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+0+2+4], v[vgprG2LB+0+2+4], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v207, v207, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v208, v208, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v209, v209, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+0+3+4], v[vgprG2LB+0+3+4], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v210, v210, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v211, v211, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v212, v212, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v201, 0x8, v201                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0+4], v[vgprG2LB+0+0+4], v201     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+0+0+4], v[vgprG2LB+0+0+4], v202     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v203, 0x8, v203                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0+4], v[vgprG2LB+0+0+4], v203     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v204, 0x8, v204                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1+4], v[vgprG2LB+0+1+4], v204     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+0+1+4], v[vgprG2LB+0+1+4], v205     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v206, 0x8, v206                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1+4], v[vgprG2LB+0+1+4], v206     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v207, 0x8, v207                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2+4], v[vgprG2LB+0+2+4], v207     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+0+2+4], v[vgprG2LB+0+2+4], v208     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v209, 0x8, v209                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2+4], v[vgprG2LB+0+2+4], v209     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v210, 0x8, v210                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3+4], v[vgprG2LB+0+3+4], v210     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+0+3+4], v[vgprG2LB+0+3+4], v211     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v212, 0x8, v212                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3+4], v[vgprG2LB+0+3+4], v212     // pack a sub 8-bit with dest


// 3. v[vgprG2LB+0+8] ~ v[vgprG2LB+3+8]
v_mov_b32 v[vgprG2LB+0+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+0+8], 4, v[vgprG2LB+0+8]      // 
v_add_co_u32 v[vgprG2LB+0+8], vcc, 0, v[vgprG2LB+0+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+0+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+0+8], vcc, 0, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v[vgprG2LB+0+8], -0x1, v[vgprG2LB+0+8], s[64:65] // 
v_mov_b32 v201, v219                    // 
v_lshlrev_b32 v201, 4, v201                          // 
v_add_co_u32 v201, vcc, 1, v201                      // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]    // 
v_add_co_u32 v201, vcc, 1, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]             // 
v_mov_b32 v202, v219                    // 
v_lshlrev_b32 v202, 4, v202                          // 
v_add_co_u32 v202, vcc, 2, v202                      // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]    // 
v_add_co_u32 v202, vcc, 2, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]             // 
v_mov_b32 v203, v219                    // 
v_lshlrev_b32 v203, 4, v203                          // 
v_add_co_u32 v203, vcc, 3, v203                      // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]    // 
v_add_co_u32 v203, vcc, 3, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]             // 
v_mov_b32 v[vgprG2LB+1+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+1+8], 4, v[vgprG2LB+1+8]      // 
v_add_co_u32 v[vgprG2LB+1+8], vcc, 4, v[vgprG2LB+1+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+1+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+1+8], vcc, 4, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v[vgprG2LB+1+8], -0x1, v[vgprG2LB+1+8], s[64:65] // 
v_mov_b32 v204, v219                    // 
v_lshlrev_b32 v204, 4, v204                          // 
v_add_co_u32 v204, vcc, 5, v204                      // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]    // 
v_add_co_u32 v204, vcc, 5, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]             // 
v_mov_b32 v205, v219                    // 
v_lshlrev_b32 v205, 4, v205                          // 
v_add_co_u32 v205, vcc, 6, v205                      // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]    // 
v_add_co_u32 v205, vcc, 6, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]             // 
v_mov_b32 v206, v219                    // 
v_lshlrev_b32 v206, 4, v206                          // 
v_add_co_u32 v206, vcc, 7, v206                      // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]    // 
v_add_co_u32 v206, vcc, 7, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]             // 
v_mov_b32 v[vgprG2LB+2+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+2+8], 4, v[vgprG2LB+2+8]      // 
v_add_co_u32 v[vgprG2LB+2+8], vcc, 8, v[vgprG2LB+2+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+2+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+2+8], vcc, 8, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v[vgprG2LB+2+8], -0x1, v[vgprG2LB+2+8], s[64:65] // 
v_mov_b32 v207, v219                    // 
v_lshlrev_b32 v207, 4, v207                          // 
v_add_co_u32 v207, vcc, 9, v207                      // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]    // 
v_add_co_u32 v207, vcc, 9, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]             // 
v_mov_b32 v208, v219                    // 
v_lshlrev_b32 v208, 4, v208                          // 
v_add_co_u32 v208, vcc, 10, v208                     // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]    // 
v_add_co_u32 v208, vcc, 10, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]             // 
v_mov_b32 v209, v219                    // 
v_lshlrev_b32 v209, 4, v209                          // 
v_add_co_u32 v209, vcc, 11, v209                     // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]    // 
v_add_co_u32 v209, vcc, 11, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]             // 
v_mov_b32 v[vgprG2LB+3+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+3+8], 4, v[vgprG2LB+3+8]      // 
v_add_co_u32 v[vgprG2LB+3+8], vcc, 12, v[vgprG2LB+3+8] // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+3+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+3+8], vcc, 12, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v[vgprG2LB+3+8], -0x1, v[vgprG2LB+3+8], s[64:65] // 
v_mov_b32 v210, v219                    // 
v_lshlrev_b32 v210, 4, v210                          // 
v_add_co_u32 v210, vcc, 13, v210                     // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]    // 
v_add_co_u32 v210, vcc, 13, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]             // 
v_mov_b32 v211, v219                    // 
v_lshlrev_b32 v211, 4, v211                          // 
v_add_co_u32 v211, vcc, 14, v211                     // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]    // 
v_add_co_u32 v211, vcc, 14, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]             // 
v_mov_b32 v212, v219                    // 
v_lshlrev_b32 v212, 4, v212                          // 
v_add_co_u32 v212, vcc, 15, v212                     // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]    // 
v_add_co_u32 v212, vcc, 15, v[vgprGlobalReadOffsetB+2] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]             // 

/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+0+0+8], v[vgprG2LB+0+0+8], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v201, v201, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v202, v202, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v203, v203, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+0+1+8], v[vgprG2LB+0+1+8], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v204, v204, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v205, v205, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v206, v206, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+0+2+8], v[vgprG2LB+0+2+8], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v207, v207, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v208, v208, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v209, v209, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+0+3+8], v[vgprG2LB+0+3+8], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v210, v210, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v211, v211, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v212, v212, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v201, 0x8, v201                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0+8], v[vgprG2LB+0+0+8], v201     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+0+0+8], v[vgprG2LB+0+0+8], v202     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v203, 0x8, v203                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0+8], v[vgprG2LB+0+0+8], v203     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v204, 0x8, v204                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1+8], v[vgprG2LB+0+1+8], v204     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+0+1+8], v[vgprG2LB+0+1+8], v205     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v206, 0x8, v206                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1+8], v[vgprG2LB+0+1+8], v206     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v207, 0x8, v207                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2+8], v[vgprG2LB+0+2+8], v207     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+0+2+8], v[vgprG2LB+0+2+8], v208     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v209, 0x8, v209                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2+8], v[vgprG2LB+0+2+8], v209     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v210, 0x8, v210                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3+8], v[vgprG2LB+0+3+8], v210     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+0+3+8], v[vgprG2LB+0+3+8], v211     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v212, 0x8, v212                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3+8], v[vgprG2LB+0+3+8], v212     // pack a sub 8-bit with dest


// 4. v[vgprG2LB+0+12] ~ v[vgprG2LB+3+12]
v_mov_b32 v[vgprG2LB+0+4+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+0+4+8], 4, v[vgprG2LB+0+4+8]      // 
v_add_co_u32 v[vgprG2LB+0+4+8], vcc, 64, v[vgprG2LB+0+4+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+0+4+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+0+4+8], vcc, 0, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v[vgprG2LB+0+4+8], -0x1, v[vgprG2LB+0+4+8], s[64:65] // 
v_mov_b32 v201, v219                    // 
v_lshlrev_b32 v201, 4, v201                          // 
v_add_co_u32 v201, vcc, 65, v201                      // 
v_cmp_lt_u32 s[64:65], v201, s[sgprLoopCounterL]    // 
v_add_co_u32 v201, vcc, 1, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v201, -0x1, v201, s[64:65]             // 
v_mov_b32 v202, v219                    // 
v_lshlrev_b32 v202, 4, v202                          // 
v_add_co_u32 v202, vcc, 66, v202                      // 
v_cmp_lt_u32 s[64:65], v202, s[sgprLoopCounterL]    // 
v_add_co_u32 v202, vcc, 2, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v202, -0x1, v202, s[64:65]             // 
v_mov_b32 v203, v219                    // 
v_lshlrev_b32 v203, 4, v203                          // 
v_add_co_u32 v203, vcc, 67, v203                      // 
v_cmp_lt_u32 s[64:65], v203, s[sgprLoopCounterL]    // 
v_add_co_u32 v203, vcc, 3, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v203, -0x1, v203, s[64:65]             // 
v_mov_b32 v[vgprG2LB+1+4+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+1+4+8], 4, v[vgprG2LB+1+4+8]      // 
v_add_co_u32 v[vgprG2LB+1+4+8], vcc, 68, v[vgprG2LB+1+4+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+1+4+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+1+4+8], vcc, 4, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v[vgprG2LB+1+4+8], -0x1, v[vgprG2LB+1+4+8], s[64:65] // 
v_mov_b32 v204, v219                    // 
v_lshlrev_b32 v204, 4, v204                          // 
v_add_co_u32 v204, vcc, 69, v204                      // 
v_cmp_lt_u32 s[64:65], v204, s[sgprLoopCounterL]    // 
v_add_co_u32 v204, vcc, 5, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v204, -0x1, v204, s[64:65]             // 
v_mov_b32 v205, v219                    // 
v_lshlrev_b32 v205, 4, v205                          // 
v_add_co_u32 v205, vcc, 70, v205                      // 
v_cmp_lt_u32 s[64:65], v205, s[sgprLoopCounterL]    // 
v_add_co_u32 v205, vcc, 6, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v205, -0x1, v205, s[64:65]             // 
v_mov_b32 v206, v219                    // 
v_lshlrev_b32 v206, 4, v206                          // 
v_add_co_u32 v206, vcc, 71, v206                      // 
v_cmp_lt_u32 s[64:65], v206, s[sgprLoopCounterL]    // 
v_add_co_u32 v206, vcc, 7, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v206, -0x1, v206, s[64:65]             // 
v_mov_b32 v[vgprG2LB+2+4+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+2+4+8], 4, v[vgprG2LB+2+4+8]      // 
v_add_co_u32 v[vgprG2LB+2+4+8], vcc, 72, v[vgprG2LB+2+4+8]  // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+2+4+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+2+4+8], vcc, 8, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v[vgprG2LB+2+4+8], -0x1, v[vgprG2LB+2+4+8], s[64:65] // 
v_mov_b32 v207, v219                    // 
v_lshlrev_b32 v207, 4, v207                          // 
v_add_co_u32 v207, vcc, 73, v207                      // 
v_cmp_lt_u32 s[64:65], v207, s[sgprLoopCounterL]    // 
v_add_co_u32 v207, vcc, 9, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v207, -0x1, v207, s[64:65]             // 
v_mov_b32 v208, v219                    // 
v_lshlrev_b32 v208, 4, v208                          // 
v_add_co_u32 v208, vcc, 74, v208                     // 
v_cmp_lt_u32 s[64:65], v208, s[sgprLoopCounterL]    // 
v_add_co_u32 v208, vcc, 10, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v208, -0x1, v208, s[64:65]             // 
v_mov_b32 v209, v219                    // 
v_lshlrev_b32 v209, 4, v209                          // 
v_add_co_u32 v209, vcc, 75, v209                     // 
v_cmp_lt_u32 s[64:65], v209, s[sgprLoopCounterL]    // 
v_add_co_u32 v209, vcc, 11, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v209, -0x1, v209, s[64:65]             // 
v_mov_b32 v[vgprG2LB+3+4+8], v219          // 
v_lshlrev_b32 v[vgprG2LB+3+4+8], 4, v[vgprG2LB+3+4+8]      // 
v_add_co_u32 v[vgprG2LB+3+4+8], vcc, 76, v[vgprG2LB+3+4+8] // 
v_cmp_lt_u32 s[64:65], v[vgprG2LB+3+4+8], s[sgprLoopCounterL] // 
v_add_co_u32 v[vgprG2LB+3+4+8], vcc, 12, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v[vgprG2LB+3+4+8], -0x1, v[vgprG2LB+3+4+8], s[64:65] // 
v_mov_b32 v210, v219                    // 
v_lshlrev_b32 v210, 4, v210                          // 
v_add_co_u32 v210, vcc, 77, v210                     // 
v_cmp_lt_u32 s[64:65], v210, s[sgprLoopCounterL]    // 
v_add_co_u32 v210, vcc, 13, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v210, -0x1, v210, s[64:65]             // 
v_mov_b32 v211, v219                    // 
v_lshlrev_b32 v211, 4, v211                          // 
v_add_co_u32 v211, vcc, 78, v211                     // 
v_cmp_lt_u32 s[64:65], v211, s[sgprLoopCounterL]    // 
v_add_co_u32 v211, vcc, 14, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v211, -0x1, v211, s[64:65]             // 
v_mov_b32 v212, v219                    // 
v_lshlrev_b32 v212, 4, v212                          // 
v_add_co_u32 v212, vcc, 79, v212                     // 
v_cmp_lt_u32 s[64:65], v212, s[sgprLoopCounterL]    // 
v_add_co_u32 v212, vcc, 15, v[vgprGlobalReadOffsetB+3] // 
v_cndmask_b32 v212, -0x1, v212, s[64:65]             // 

/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+0+0+4+8], v[vgprG2LB+0+0+4+8], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v201, v201, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v202, v202, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v203, v203, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+0+1+4+8], v[vgprG2LB+0+1+4+8], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v204, v204, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v205, v205, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v206, v206, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+0+2+4+8], v[vgprG2LB+0+2+4+8], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v207, v207, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v208, v208, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v209, v209, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+0+3+4+8], v[vgprG2LB+0+3+4+8], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v210, v210, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v211, v211, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v212, v212, s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v201, 0x8, v201                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0+4+8], v[vgprG2LB+0+0+4+8], v201     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+0+0+4+8], v[vgprG2LB+0+0+4+8], v202     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v203, 0x8, v203                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0+4+8], v[vgprG2LB+0+0+4+8], v203     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v204, 0x8, v204                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1+4+8], v[vgprG2LB+0+1+4+8], v204     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+0+1+4+8], v[vgprG2LB+0+1+4+8], v205     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v206, 0x8, v206                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1+4+8], v[vgprG2LB+0+1+4+8], v206     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v207, 0x8, v207                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2+4+8], v[vgprG2LB+0+2+4+8], v207     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+0+2+4+8], v[vgprG2LB+0+2+4+8], v208     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v209, 0x8, v209                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2+4+8], v[vgprG2LB+0+2+4+8], v209     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v210, 0x8, v210                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3+4+8], v[vgprG2LB+0+3+4+8], v210     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+0+3+4+8], v[vgprG2LB+0+3+4+8], v211     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v212, 0x8, v212                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3+4+8], v[vgprG2LB+0+3+4+8], v212     // pack a sub 8-bit with dest

s_add_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 127 // tail add
s_lshr_b32 s[sgprLoopCounterL], s[sgprLoopCounterL], 7  // 

/* Recalc local read offsets */


s_waitcnt vmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_barrier //

v_mov_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+1], v[vgprG2LB+0:vgprG2LB+1]
v_mov_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+3], v[vgprG2LB+2:vgprG2LB+3]
v_mov_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+5], v[vgprG2LB+4:vgprG2LB+5]
v_mov_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+7], v[vgprG2LB+6:vgprG2LB+7]
v_mov_b64 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+9], v[vgprG2LB+8:vgprG2LB+9]
v_mov_b64 v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+11], v[vgprG2LB+10:vgprG2LB+11]
v_mov_b64 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+13], v[vgprG2LB+12:vgprG2LB+13]
v_mov_b64 v[vgprValuB_X0_I0+14:vgprValuB_X0_I0+15], v[vgprG2LB+14:vgprG2LB+15]

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_barrier //


/* TailLoop: Check VGPR.checkin for INT8 LW */


/* local read reset offsets a */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA], 0x7fff, v[vgprLocalReadAddrA] // reset Red,Blk -> Red


/* local read reset offsets b */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB], 0x7fff, v[vgprLocalReadAddrB] // reset Red,Blk -> Red

/* tail loop: macs */

TailLoopBeginL_5:

/* local read a */

ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3],   v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7],   v[vgprLocalReadAddrA] offset:0x800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=1 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11],  v[vgprLocalReadAddrA] offset:0x1000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=2 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:0x1800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=3 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0x2000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=4 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:0x2800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=5 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0x3000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=6 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:0x3800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=7 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0x4000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=8 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:0x4800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=9 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0x5000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=10 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:0x5800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=11 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0x6000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=12 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:0x6800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=13 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0x7000 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=14 rIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:0x7800 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=15 rIdx=0 buffer=1 iui=0

s_waitcnt lgkmcnt(0)
MAC_32x4x2_X0_I0
s_barrier


ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0x400
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:0xc00
ds_read_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+11], v[vgprLocalReadAddrA] offset:0x1400
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+15], v[vgprLocalReadAddrA] offset:0x1c00
ds_read_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+19], v[vgprLocalReadAddrA] offset:0x2400
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+23], v[vgprLocalReadAddrA] offset:0x2c00
ds_read_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+27], v[vgprLocalReadAddrA] offset:0x3400
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+31], v[vgprLocalReadAddrA] offset:0x3c00
ds_read_b128 v[vgprValuA_X0_I0+32:vgprValuA_X0_I0+35], v[vgprLocalReadAddrA] offset:0x4400
ds_read_b128 v[vgprValuA_X0_I0+36:vgprValuA_X0_I0+39], v[vgprLocalReadAddrA] offset:0x4c00
ds_read_b128 v[vgprValuA_X0_I0+40:vgprValuA_X0_I0+43], v[vgprLocalReadAddrA] offset:0x5400
ds_read_b128 v[vgprValuA_X0_I0+44:vgprValuA_X0_I0+47], v[vgprLocalReadAddrA] offset:0x5c00
ds_read_b128 v[vgprValuA_X0_I0+48:vgprValuA_X0_I0+51], v[vgprLocalReadAddrA] offset:0x6400
ds_read_b128 v[vgprValuA_X0_I0+52:vgprValuA_X0_I0+55], v[vgprLocalReadAddrA] offset:0x6c00
ds_read_b128 v[vgprValuA_X0_I0+56:vgprValuA_X0_I0+59], v[vgprLocalReadAddrA] offset:0x7400
ds_read_b128 v[vgprValuA_X0_I0+60:vgprValuA_X0_I0+63], v[vgprLocalReadAddrA] offset:0x7c00
s_waitcnt lgkmcnt(0)
MAC_32x4x2_X0_I1
s_barrier

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_5                    // restart LoopL

SkipTailLoopL_7:

Summation_End_23:
/* endSummation: add vgpr [128...198) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
.set StrideSwizzleB, UNDEF
.set StructNumB, UNDEF
.set StructBitB, UNDEF
.set StructBit1B, UNDEF
.set StrideStructB, UNDEF
.set OffsetStructB, UNDEF
.set ShadowLimitB_forTailLoop, UNDEF

s_barrier
; ValuC_INT32_TO_FP32
SMQUANT
/***********scale A read*******/

label_BiasAddrValid_End_1:

/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_and_b32 v128, 63, v[vgprSerial]                  // v0 = v[vgprSerial] & 63
v_lshrrev_b32 v129, 6, v[vgprSerial]               // v4 = v[vgprSerial] / 64
v_and_b32 v132, 15, v128                           // v5 = v0 & 15
v_lshrrev_b32 v133, 4, v128                        // v3 = v0 / 16
v_and_b32 v134, 0, v129                            // v4&1
s_mov_b32 s54, 128                                 //  
v_mul_lo_u32 v134, s54, v134                       //  
v_lshlrev_b32 v128, 0, v132                        // ((v[vgprSerial]&63)&15)*2
_v_add_co_u32 v128, vcc, v128, v134                // row
v_lshrrev_b32 v134, 0, v129                        // v4>>1
s_mov_b32 s54, 32                                  //  
v_mul_lo_u32 v134, s54, v134                       //  
v_lshlrev_b32 v129, 0, v133                        // ((v[vgprSerial]&63)&15)*2
_v_add_co_u32 v129, vcc, v129, v134                // col
s_mul_i32 s[sgprStrideoffC+0], 8, s[sgprStridesC] //
s_mul_i32 s[sgprStrideoffC+1], 16, s[sgprStridesC] //
s_mul_i32 s[sgprStrideoffC+2], 24, s[sgprStridesC] //
s_mul_i32 s[sgprStrideoffD+0], 8, s[sgprStridesD] // 
s_mul_i32 s[sgprStrideoffD+1], 16, s[sgprStridesD] // 
s_mul_i32 s[sgprStrideoffD+2], 24, s[sgprStridesD] // 
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          // rowStart vgpr
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          // rowStart vgpr

s_mul_i32 s54, 0x100, s[sgprWorkGroup0]            // s54 = wg0*MT0
_v_add_co_u32 v128, vcc, s54, v128                 // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s56, 0x100, s[sgprWorkGroup1]            // <- wg1*MT1
_v_add_co_u32 v129, vcc, s56, v129                 // coord1 = tid1*VW + wg1*MT1


/* not-LocalSplitU: global write */

s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 GW_Beta_38                          // Branch if Beta is not zero

s_and_b32 s54, 255, s[sgprSizeI]                   // s54 = s[sgprSizeI] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_29                         // jump if edges required
s_and_b32 s54, 255, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_29                         // jump if edges required
GW_B0_E0_26:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=128 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1); (0,8,0,0:vw1); (0,9,0,0:vw1); (0,10,0,0:vw1); (0,11,0,0:vw1); (0,12,0,0:vw1); (0,13,0,0:vw1); (0,14,0,0:vw1); (0,15,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (1,5,0,0:vw1); (1,6,0,0:vw1); (1,7,0,0:vw1); (1,8,0,0:vw1); (1,9,0,0:vw1); (1,10,0,0:vw1); (1,11,0,0:vw1); (1,12,0,0:vw1); (1,13,0,0:vw1); (1,14,0,0:vw1); (1,15,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (2,5,0,0:vw1); (2,6,0,0:vw1); (2,7,0,0:vw1); (2,8,0,0:vw1); (2,9,0,0:vw1); (2,10,0,0:vw1); (2,11,0,0:vw1); (2,12,0,0:vw1); (2,13,0,0:vw1); (2,14,0,0:vw1); (2,15,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (3,5,0,0:vw1); (3,6,0,0:vw1); (3,7,0,0:vw1); (3,8,0,0:vw1); (3,9,0,0:vw1); (3,10,0,0:vw1); (3,11,0,0:vw1); (3,12,0,0:vw1); (3,13,0,0:vw1); (3,14,0,0:vw1); (3,15,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (4,5,0,0:vw1); (4,6,0,0:vw1); (4,7,0,0:vw1); (4,8,0,0:vw1); (4,9,0,0:vw1); (4,10,0,0:vw1); (4,11,0,0:vw1); (4,12,0,0:vw1); (4,13,0,0:vw1); (4,14,0,0:vw1); (4,15,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (5,5,0,0:vw1); (5,6,0,0:vw1); (5,7,0,0:vw1); (5,8,0,0:vw1); (5,9,0,0:vw1); (5,10,0,0:vw1); (5,11,0,0:vw1); (5,12,0,0:vw1); (5,13,0,0:vw1); (5,14,0,0:vw1); (5,15,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (6,5,0,0:vw1); (6,6,0,0:vw1); (6,7,0,0:vw1); (6,8,0,0:vw1); (6,9,0,0:vw1); (6,10,0,0:vw1); (6,11,0,0:vw1); (6,12,0,0:vw1); (6,13,0,0:vw1); (6,14,0,0:vw1); (6,15,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1); (7,5,0,0:vw1); (7,6,0,0:vw1); (7,7,0,0:vw1); (7,8,0,0:vw1); (7,9,0,0:vw1); (7,10,0,0:vw1); (7,11,0,0:vw1); (7,12,0,0:vw1); (7,13,0,0:vw1); (7,14,0,0:vw1); (7,15,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
/* (d1,vc1,d0,vc0)=(0,0,8,0) */
/* (d1,vc1,d0,vc0)=(0,0,9,0) */
/* (d1,vc1,d0,vc0)=(0,0,10,0) */
/* (d1,vc1,d0,vc0)=(0,0,11,0) */
/* (d1,vc1,d0,vc0)=(0,0,12,0) */
/* (d1,vc1,d0,vc0)=(0,0,13,0) */
/* (d1,vc1,d0,vc0)=(0,0,14,0) */
/* (d1,vc1,d0,vc0)=(0,0,15,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
/* (d1,vc1,d0,vc0)=(1,0,8,0) */
/* (d1,vc1,d0,vc0)=(1,0,9,0) */
/* (d1,vc1,d0,vc0)=(1,0,10,0) */
/* (d1,vc1,d0,vc0)=(1,0,11,0) */
/* (d1,vc1,d0,vc0)=(1,0,12,0) */
/* (d1,vc1,d0,vc0)=(1,0,13,0) */
/* (d1,vc1,d0,vc0)=(1,0,14,0) */
/* (d1,vc1,d0,vc0)=(1,0,15,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
/* (d1,vc1,d0,vc0)=(2,0,5,0) */
/* (d1,vc1,d0,vc0)=(2,0,6,0) */
/* (d1,vc1,d0,vc0)=(2,0,7,0) */
/* (d1,vc1,d0,vc0)=(2,0,8,0) */
/* (d1,vc1,d0,vc0)=(2,0,9,0) */
/* (d1,vc1,d0,vc0)=(2,0,10,0) */
/* (d1,vc1,d0,vc0)=(2,0,11,0) */
/* (d1,vc1,d0,vc0)=(2,0,12,0) */
/* (d1,vc1,d0,vc0)=(2,0,13,0) */
/* (d1,vc1,d0,vc0)=(2,0,14,0) */
/* (d1,vc1,d0,vc0)=(2,0,15,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
/* (d1,vc1,d0,vc0)=(3,0,5,0) */
/* (d1,vc1,d0,vc0)=(3,0,6,0) */
/* (d1,vc1,d0,vc0)=(3,0,7,0) */
/* (d1,vc1,d0,vc0)=(3,0,8,0) */
/* (d1,vc1,d0,vc0)=(3,0,9,0) */
/* (d1,vc1,d0,vc0)=(3,0,10,0) */
/* (d1,vc1,d0,vc0)=(3,0,11,0) */
/* (d1,vc1,d0,vc0)=(3,0,12,0) */
/* (d1,vc1,d0,vc0)=(3,0,13,0) */
/* (d1,vc1,d0,vc0)=(3,0,14,0) */
/* (d1,vc1,d0,vc0)=(3,0,15,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
/* (d1,vc1,d0,vc0)=(4,0,5,0) */
/* (d1,vc1,d0,vc0)=(4,0,6,0) */
/* (d1,vc1,d0,vc0)=(4,0,7,0) */
/* (d1,vc1,d0,vc0)=(4,0,8,0) */
/* (d1,vc1,d0,vc0)=(4,0,9,0) */
/* (d1,vc1,d0,vc0)=(4,0,10,0) */
/* (d1,vc1,d0,vc0)=(4,0,11,0) */
/* (d1,vc1,d0,vc0)=(4,0,12,0) */
/* (d1,vc1,d0,vc0)=(4,0,13,0) */
/* (d1,vc1,d0,vc0)=(4,0,14,0) */
/* (d1,vc1,d0,vc0)=(4,0,15,0) */
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
/* (d1,vc1,d0,vc0)=(5,0,4,0) */
/* (d1,vc1,d0,vc0)=(5,0,5,0) */
/* (d1,vc1,d0,vc0)=(5,0,6,0) */
/* (d1,vc1,d0,vc0)=(5,0,7,0) */
/* (d1,vc1,d0,vc0)=(5,0,8,0) */
/* (d1,vc1,d0,vc0)=(5,0,9,0) */
/* (d1,vc1,d0,vc0)=(5,0,10,0) */
/* (d1,vc1,d0,vc0)=(5,0,11,0) */
/* (d1,vc1,d0,vc0)=(5,0,12,0) */
/* (d1,vc1,d0,vc0)=(5,0,13,0) */
/* (d1,vc1,d0,vc0)=(5,0,14,0) */
/* (d1,vc1,d0,vc0)=(5,0,15,0) */
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
/* (d1,vc1,d0,vc0)=(6,0,5,0) */
/* (d1,vc1,d0,vc0)=(6,0,6,0) */
/* (d1,vc1,d0,vc0)=(6,0,7,0) */
/* (d1,vc1,d0,vc0)=(6,0,8,0) */
/* (d1,vc1,d0,vc0)=(6,0,9,0) */
/* (d1,vc1,d0,vc0)=(6,0,10,0) */
/* (d1,vc1,d0,vc0)=(6,0,11,0) */
/* (d1,vc1,d0,vc0)=(6,0,12,0) */
/* (d1,vc1,d0,vc0)=(6,0,13,0) */
/* (d1,vc1,d0,vc0)=(6,0,14,0) */
/* (d1,vc1,d0,vc0)=(6,0,15,0) */
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
/* (d1,vc1,d0,vc0)=(7,0,5,0) */
/* (d1,vc1,d0,vc0)=(7,0,6,0) */
/* (d1,vc1,d0,vc0)=(7,0,7,0) */
/* (d1,vc1,d0,vc0)=(7,0,8,0) */
/* (d1,vc1,d0,vc0)=(7,0,9,0) */
/* (d1,vc1,d0,vc0)=(7,0,10,0) */
/* (d1,vc1,d0,vc0)=(7,0,11,0) */
/* (d1,vc1,d0,vc0)=(7,0,12,0) */
/* (d1,vc1,d0,vc0)=(7,0,13,0) */
/* (d1,vc1,d0,vc0)=(7,0,14,0) */
/* (d1,vc1,d0,vc0)=(7,0,15,0) */
_v_add_lshl_u32 v134, v131, v128, 0x1              // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128


/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 8, 0, 0), (0, 9, 0, 0), (0, 10, 0, 0), (0, 11, 0, 0), (0, 12, 0, 0), (0, 13, 0, 0), (0, 14, 0, 0), (0, 15, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0), (1, 8, 0, 0), (1, 9, 0, 0), (1, 10, 0, 0), (1, 11, 0, 0), (1, 12, 0, 0), (1, 13, 0, 0), (1, 14, 0, 0), (1, 15, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (2, 5, 0, 0), (2, 6, 0, 0), (2, 7, 0, 0), (2, 8, 0, 0), (2, 9, 0, 0), (2, 10, 0, 0), (2, 11, 0, 0), (2, 12, 0, 0), (2, 13, 0, 0), (2, 14, 0, 0), (2, 15, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (3, 5, 0, 0), (3, 6, 0, 0), (3, 7, 0, 0), (3, 8, 0, 0), (3, 9, 0, 0), (3, 10, 0, 0), (3, 11, 0, 0), (3, 12, 0, 0), (3, 13, 0, 0), (3, 14, 0, 0), (3, 15, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (4, 5, 0, 0), (4, 6, 0, 0), (4, 7, 0, 0), (4, 8, 0, 0), (4, 9, 0, 0), (4, 10, 0, 0), (4, 11, 0, 0), (4, 12, 0, 0), (4, 13, 0, 0), (4, 14, 0, 0), (4, 15, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (5, 4, 0, 0), (5, 5, 0, 0), (5, 6, 0, 0), (5, 7, 0, 0), (5, 8, 0, 0), (5, 9, 0, 0), (5, 10, 0, 0), (5, 11, 0, 0), (5, 12, 0, 0), (5, 13, 0, 0), (5, 14, 0, 0), (5, 15, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (6, 5, 0, 0), (6, 6, 0, 0), (6, 7, 0, 0), (6, 8, 0, 0), (6, 9, 0, 0), (6, 10, 0, 0), (6, 11, 0, 0), (6, 12, 0, 0), (6, 13, 0, 0), (6, 14, 0, 0), (6, 15, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0), (7, 5, 0, 0), (7, 6, 0, 0), (7, 7, 0, 0), (7, 8, 0, 0), (7, 9, 0, 0), (7, 10, 0, 0), (7, 11, 0, 0), (7, 12, 0, 0), (7, 13, 0, 0), (7, 14, 0, 0), (7, 15, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha


label_BiasAddrValid_End_1_1:
FP32_TO_BF16_ALL 0

/* apply mask, calc new C and issue writes */
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v1, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v2, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v3, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v4, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v5, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v6, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v7, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v8, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v9, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v10, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v11, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v12, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v13, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v14, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v15, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v16, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v17, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v18, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v19, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v21, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v22, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v23, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v24, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v25, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v26, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v27, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v28, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v29, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v30, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v31, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                //
buffer_store_short v32, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v33, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v34, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v35, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v36, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v37, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v38, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v39, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v41, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v42, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v43, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v44, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v45, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v46, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v47, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v48, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v49, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v50, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v51, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v52, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v53, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v54, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v55, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v56, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v57, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v58, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v59, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v61, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v62, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v63, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_lshl_u32 v134, v131, v128, 0x1              // 
s_mul_i32 s54, s[sgprStrideD1J], 32                // scale StrideD *= numRows(16) * bpe

s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v64, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v65, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v66, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v67, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v68, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v69, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v70, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v71, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v72, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v73, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v74, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v75, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v76, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v77, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v78, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v79, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v81, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v82, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v83, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v84, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v85, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v86, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v87, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v88, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v89, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v90, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v91, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v92, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v93, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v94, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v95, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v96, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v97, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v98, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v99, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v101, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v102, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v103, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v104, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v106, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v107, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v108, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v109, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v110, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v111, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v112, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v113, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v114, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v115, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v116, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v117, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v118, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v119, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v121, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v122, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v123, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20                // 
buffer_store_short v124, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v125, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v127, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
s_branch label_GW_End_37                           // jump to end



GW_B0_E1_29:

/* edge=1, allocate 48 sgpr. perBatch=6 perElement=2 elementsPerBatch=21 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1); (0,8,0,0:vw1); (0,9,0,0:vw1); (0,10,0,0:vw1); (0,11,0,0:vw1); (0,12,0,0:vw1); (0,13,0,0:vw1); (0,14,0,0:vw1); (0,15,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1) */
/******************************************/

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 8, 0, 0), (0, 9, 0, 0), (0, 10, 0, 0), (0, 11, 0, 0), (0, 12, 0, 0), (0, 13, 0, 0), (0, 14, 0, 0), (0, 15, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha

/* rC *= alpha batchEements=[(1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0), (1, 8, 0, 0), (1, 9, 0, 0), (1, 10, 0, 0), (1, 11, 0, 0), (1, 12, 0, 0), (1, 13, 0, 0), (1, 14, 0, 0), (1, 15, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (2, 5, 0, 0), (2, 6, 0, 0), (2, 7, 0, 0), (2, 8, 0, 0), (2, 9, 0, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha

/* rC *= alpha batchEements=[(2, 10, 0, 0), (2, 11, 0, 0), (2, 12, 0, 0), (2, 13, 0, 0), (2, 14, 0, 0), (2, 15, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (3, 5, 0, 0), (3, 6, 0, 0), (3, 7, 0, 0), (3, 8, 0, 0), (3, 9, 0, 0), (3, 10, 0, 0), (3, 11, 0, 0), (3, 12, 0, 0), (3, 13, 0, 0), (3, 14, 0, 0)] */
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha

/* rC *= alpha batchEements=[(3, 15, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (4, 5, 0, 0), (4, 6, 0, 0), (4, 7, 0, 0), (4, 8, 0, 0), (4, 9, 0, 0), (4, 10, 0, 0), (4, 11, 0, 0), (4, 12, 0, 0), (4, 13, 0, 0), (4, 14, 0, 0), (4, 15, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha

/* rC *= alpha batchEements=[(5, 4, 0, 0), (5, 5, 0, 0), (5, 6, 0, 0), (5, 7, 0, 0), (5, 8, 0, 0), (5, 9, 0, 0), (5, 10, 0, 0), (5, 11, 0, 0), (5, 12, 0, 0), (5, 13, 0, 0), (5, 14, 0, 0), (5, 15, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (6, 5, 0, 0), (6, 6, 0, 0), (6, 7, 0, 0), (6, 8, 0, 0)] */
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha


/* rC *= alpha batchEements=[(6, 9, 0, 0), (6, 10, 0, 0), (6, 11, 0, 0), (6, 12, 0, 0), (6, 13, 0, 0), (6, 14, 0, 0), (6, 15, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0), (7, 5, 0, 0), (7, 6, 0, 0), (7, 7, 0, 0), (7, 8, 0, 0), (7, 9, 0, 0), (7, 10, 0, 0), (7, 11, 0, 0), (7, 12, 0, 0), (7, 13, 0, 0)] */
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha

/* rC *= alpha batchEements=[(7, 14, 0, 0), (7, 15, 0, 0)] */
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha


label_BiasAddrValid_End_1_2:

FP32_TO_BF16_ALL 0

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,8,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,9,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+0], v143  // add 4* loop offset
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,10,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+1], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,11,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+2], v145  // add 4* loop offset
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,12,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,13,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+0], v147  // add 4* loop offset
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,14,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+1], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,15,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+2], v149  // add 4* loop offset
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+0], v151  // add 4* loop offset
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+1], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+2], v153  // add 4* loop offset
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=4 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[100:101], v129, s[sgprSizeJ]        // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, -1, v154, s[100:101]           // LDD clip if OOB. offset


/* apply mask, calc new C and issue writes */
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v1, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v2, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v3, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v4, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v5, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v6, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v7, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v8, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v9, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v10, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v11, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v12, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v13, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v14, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v15, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v16, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v17, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v18, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v19, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v20, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (1,5,0,0:vw1); (1,6,0,0:vw1); (1,7,0,0:vw1); (1,8,0,0:vw1); (1,9,0,0:vw1); (1,10,0,0:vw1); (1,11,0,0:vw1); (1,12,0,0:vw1); (1,13,0,0:vw1); (1,14,0,0:vw1); (1,15,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (2,5,0,0:vw1); (2,6,0,0:vw1); (2,7,0,0:vw1); (2,8,0,0:vw1); (2,9,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=5 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprStrideoffD+0], v134  // add 4* loop offset
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=6 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+1], v135  // add 4* loop offset
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=7 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+2], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,8,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=8 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,9,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=9 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+0], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,10,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=10 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+1], v139  // add 4* loop offset
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,11,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=11 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,12,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=12 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,13,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=13 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprStrideoffD+0], v142  // add 4* loop offset
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,14,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=14 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+1], v143  // add 4* loop offset
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,15,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+2], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+0], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+1], v147  // add 4* loop offset
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,5,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprStrideoffD+0], v150  // add 4* loop offset
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,6,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+1], v151  // add 4* loop offset
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,7,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+2], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,8,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,9,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[100:101], v133, s[sgprSizeJ]        // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+0], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[100:101]           // LDD clip if OOB. offset


/* apply mask, calc new C and issue writes */
buffer_store_short v21, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v22, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v23, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v24, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v25, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v26, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v27, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v28, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v29, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v30, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v31, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v32, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v33, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v34, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v35, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v36, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v37, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v38, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v39, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v40, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v41, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (2,10,0,0:vw1); (2,11,0,0:vw1); (2,12,0,0:vw1); (2,13,0,0:vw1); (2,14,0,0:vw1); (2,15,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (3,5,0,0:vw1); (3,6,0,0:vw1); (3,7,0,0:vw1); (3,8,0,0:vw1); (3,9,0,0:vw1); (3,10,0,0:vw1); (3,11,0,0:vw1); (3,12,0,0:vw1); (3,13,0,0:vw1); (3,14,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,10,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprStrideoffD+1], v134  // add 4* loop offset
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,11,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+2], v135  // add 4* loop offset
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,12,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,13,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+0], v137  // add 4* loop offset
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,14,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,15,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+2], v139  // add 4* loop offset
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+0], v141  // add 4* loop offset
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprStrideoffD+1], v142  // add 4* loop offset
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+2], v143  // add 4* loop offset
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=4 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,5,0) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=5 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+0], v145  // add 4* loop offset
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,6,0) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=6 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,7,0) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=7 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+2], v147  // add 4* loop offset
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,8,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=8 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,9,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=9 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+0], v149  // add 4* loop offset
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,10,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=10 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprStrideoffD+1], v150  // add 4* loop offset
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,11,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=11 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+2], v151  // add 4* loop offset
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,12,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=12 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,13,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=13 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+0], v153  // add 4* loop offset
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,14,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=14 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[100:101], v133, s[sgprSizeJ]        // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[100:101]           // LDD clip if OOB. offset


/* apply mask, calc new C and issue writes */
buffer_store_short v42, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v43, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v44, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v45, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v46, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v47, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v48, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v49, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v50, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v51, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v52, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v53, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v54, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v55, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v56, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v57, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v58, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v59, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v60, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v61, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v62, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (3,15,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (4,5,0,0:vw1); (4,6,0,0:vw1); (4,7,0,0:vw1); (4,8,0,0:vw1); (4,9,0,0:vw1); (4,10,0,0:vw1); (4,11,0,0:vw1); (4,12,0,0:vw1); (4,13,0,0:vw1); (4,14,0,0:vw1); (4,15,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,15,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprStrideoffD+2], v134  // add 4* loop offset
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], 32 // add offset stride

v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+1], v137  // add 4* loop offset
v_add_co_u32 v137, vcc, s[sgprEdgeCheck+1], v137   // 
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+2], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprEdgeCheck+1], v139   // 
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,5,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+0], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,6,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+1], v141  // add 4* loop offset
v_add_co_u32 v141, vcc, s[sgprEdgeCheck+1], v141   // 
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,7,0) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprStrideoffD+2], v142  // add 4* loop offset
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,8,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprEdgeCheck+1], v143   // 
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,9,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,10,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+1], v145  // add 4* loop offset
v_add_co_u32 v145, vcc, s[sgprEdgeCheck+1], v145   // 
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,11,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+2], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,12,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprEdgeCheck+1], v147   // 
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,13,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+0], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,14,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+1], v149  // add 4* loop offset
v_add_co_u32 v149, vcc, s[sgprEdgeCheck+1], v149   // 
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,15,0) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprStrideoffD+2], v150  // add 4* loop offset
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprEdgeCheck+1], v151   // 
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+1], v153  // add 4* loop offset
v_add_co_u32 v153, vcc, s[sgprEdgeCheck+1], v153   // 
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[100:101], v133, s[sgprSizeJ]        // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+2], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[100:101]           // LDD clip if OOB. offset


/* apply mask, calc new C and issue writes */
buffer_store_short v63, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v64, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v65, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v66, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v67, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v68, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v69, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v70, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v71, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v72, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v73, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v74, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v75, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v76, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v77, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v78, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v79, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v80, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v81, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v82, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v83, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (5,4,0,0:vw1); (5,5,0,0:vw1); (5,6,0,0:vw1); (5,7,0,0:vw1); (5,8,0,0:vw1); (5,9,0,0:vw1); (5,10,0,0:vw1); (5,11,0,0:vw1); (5,12,0,0:vw1); (5,13,0,0:vw1); (5,14,0,0:vw1); (5,15,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (6,5,0,0:vw1); (6,6,0,0:vw1); (6,7,0,0:vw1); (6,8,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(5,0,4,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=4 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,5,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=5 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+0], v135  // add 4* loop offset
v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,6,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=6 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+1], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,7,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=7 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprStrideoffD+2], v137  // add 4* loop offset
v_add_co_u32 v137, vcc, s[sgprEdgeCheck+1], v137   // 
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,8,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=8 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,9,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=9 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+0], v139  // add 4* loop offset
v_add_co_u32 v139, vcc, s[sgprEdgeCheck+1], v139   // 
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,10,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=10 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+1], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,11,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=11 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprStrideoffD+2], v141  // add 4* loop offset
v_add_co_u32 v141, vcc, s[sgprEdgeCheck+1], v141   // 
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,12,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=12 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,13,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=13 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+0], v143  // add 4* loop offset
v_add_co_u32 v143, vcc, s[sgprEdgeCheck+1], v143   // 
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,14,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=14 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+1], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(5,0,15,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprStrideoffD+2], v145  // add 4* loop offset
v_add_co_u32 v145, vcc, s[sgprEdgeCheck+1], v145   // 
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+0], v147  // add 4* loop offset
v_add_co_u32 v147, vcc, s[sgprEdgeCheck+1], v147   // 
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+1], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprStrideoffD+2], v149  // add 4* loop offset
v_add_co_u32 v149, vcc, s[sgprEdgeCheck+1], v149   // 
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,5,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+0], v151  // add 4* loop offset
v_add_co_u32 v151, vcc, s[sgprEdgeCheck+1], v151   // 
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,6,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+1], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,7,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprStrideoffD+2], v153  // add 4* loop offset
v_add_co_u32 v153, vcc, s[sgprEdgeCheck+1], v153   // 
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,8,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[100:101], v133, s[sgprSizeJ]        // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[100:101]           // LDD clip if OOB. offset


/* apply mask, calc new C and issue writes */
buffer_store_short v84, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v85, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v86, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v87, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v88, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v89, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v90, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v91, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v92, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v93, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v94, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v95, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v96, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v97, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v98, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v99, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v100, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v101, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v102, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v103, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v104, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (6,9,0,0:vw1); (6,10,0,0:vw1); (6,11,0,0:vw1); (6,12,0,0:vw1); (6,13,0,0:vw1); (6,14,0,0:vw1); (6,15,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1); (7,5,0,0:vw1); (7,6,0,0:vw1); (7,7,0,0:vw1); (7,8,0,0:vw1); (7,9,0,0:vw1); (7,10,0,0:vw1); (7,11,0,0:vw1); (7,12,0,0:vw1); (7,13,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(6,0,9,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprStrideoffD+0], v134  // add 4* loop offset
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,10,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+1], v135  // add 4* loop offset
v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,11,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+2], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,12,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v137, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v137, vcc, s[sgprEdgeCheck+1], v137   // 
v_cndmask_b32 v137, -1, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,13,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+0], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,14,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v139, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v139, vcc, s[sgprStrideoffD+1], v139  // add 4* loop offset
v_add_co_u32 v139, vcc, s[sgprEdgeCheck+1], v139   // 
v_cndmask_b32 v139, -1, v139, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(6,0,15,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v141, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v141, vcc, s[sgprEdgeCheck+1], v141   // 
v_cndmask_b32 v141, -1, v141, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprStrideoffD+0], v142  // add 4* loop offset
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v143, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v143, vcc, s[sgprStrideoffD+1], v143  // add 4* loop offset
v_add_co_u32 v143, vcc, s[sgprEdgeCheck+1], v143   // 
v_cndmask_b32 v143, -1, v143, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+2], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=4 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v145, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v145, vcc, s[sgprEdgeCheck+1], v145   // 
v_cndmask_b32 v145, -1, v145, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,5,0) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=5 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+0], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,6,0) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=6 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v147, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v147, vcc, s[sgprStrideoffD+1], v147  // add 4* loop offset
v_add_co_u32 v147, vcc, s[sgprEdgeCheck+1], v147   // 
v_cndmask_b32 v147, -1, v147, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,7,0) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=7 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,8,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=8 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v149, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v149, vcc, s[sgprEdgeCheck+1], v149   // 
v_cndmask_b32 v149, -1, v149, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,9,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=9 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprStrideoffD+0], v150  // add 4* loop offset
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,10,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=10 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v151, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v151, vcc, s[sgprStrideoffD+1], v151  // add 4* loop offset
v_add_co_u32 v151, vcc, s[sgprEdgeCheck+1], v151   // 
v_cndmask_b32 v151, -1, v151, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,11,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=11 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+2], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,12,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=12 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v153, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v153, vcc, s[sgprEdgeCheck+1], v153   // 
v_cndmask_b32 v153, -1, v153, s[98:99]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,13,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=13 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[100:101], v133, s[sgprSizeJ]        // coord1 < size1
s_and_b64 s[100:101], s[54:55], s[100:101]         // in0 && in1
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+0], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[100:101]           // LDD clip if OOB. offset


/* apply mask, calc new C and issue writes */
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v106, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v107, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v108, v137, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v109, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v110, v139, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v111, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v112, v141, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v113, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v114, v143, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v115, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v116, v145, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v117, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v118, v147, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v119, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v120, v149, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v121, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v122, v151, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v123, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v124, v153, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v125, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (7,14,0,0:vw1); (7,15,0,0:vw1)      */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(7,0,14,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=14 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprStrideoffD+1], v134  // add 4* loop offset
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(7,0,15,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v135, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v135, vcc, s[sgprStrideoffD+2], v135  // add 4* loop offset
v_add_co_u32 v135, vcc, s[sgprEdgeCheck+1], v135   // 
v_cndmask_b32 v135, -1, v135, s[62:63]             // LDD clip if OOB. offset



/* apply mask, calc new C and issue writes */
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v127, v135, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
s_branch label_GW_End_37                           // jump to end
GW_Beta_38:
s_and_b32 s54, 255, s[sgprSizeI]                   // s54 = s[sgprSizeI] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B1_E1_36                         // jump if edges required
s_and_b32 s54, 255, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 256
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B1_E1_36                         // jump if edges required
GW_B1_E0_33:

/* edge=0, allocate 2 sgpr. perBatch=2 perElement=0 elementsPerBatch=102 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1); (0,8,0,0:vw1); (0,9,0,0:vw1); (0,10,0,0:vw1); (0,11,0,0:vw1); (0,12,0,0:vw1); (0,13,0,0:vw1); (0,14,0,0:vw1); (0,15,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (1,4,0,0:vw1); (1,5,0,0:vw1); (1,6,0,0:vw1); (1,7,0,0:vw1); (1,8,0,0:vw1); (1,9,0,0:vw1); (1,10,0,0:vw1); (1,11,0,0:vw1); (1,12,0,0:vw1); (1,13,0,0:vw1); (1,14,0,0:vw1); (1,15,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (2,4,0,0:vw1); (2,5,0,0:vw1); (2,6,0,0:vw1); (2,7,0,0:vw1); (2,8,0,0:vw1); (2,9,0,0:vw1); (2,10,0,0:vw1); (2,11,0,0:vw1); (2,12,0,0:vw1); (2,13,0,0:vw1); (2,14,0,0:vw1); (2,15,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1); (3,4,0,0:vw1); (3,5,0,0:vw1); (3,6,0,0:vw1); (3,7,0,0:vw1); (3,8,0,0:vw1); (3,9,0,0:vw1); (3,10,0,0:vw1); (3,11,0,0:vw1); (3,12,0,0:vw1); (3,13,0,0:vw1); (3,14,0,0:vw1); (3,15,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1); (4,2,0,0:vw1); (4,3,0,0:vw1); (4,4,0,0:vw1); (4,5,0,0:vw1); (4,6,0,0:vw1); (4,7,0,0:vw1); (4,8,0,0:vw1); (4,9,0,0:vw1); (4,10,0,0:vw1); (4,11,0,0:vw1); (4,12,0,0:vw1); (4,13,0,0:vw1); (4,14,0,0:vw1); (4,15,0,0:vw1); (5,0,0,0:vw1); (5,1,0,0:vw1); (5,2,0,0:vw1); (5,3,0,0:vw1); (5,4,0,0:vw1); (5,5,0,0:vw1); (5,6,0,0:vw1); (5,7,0,0:vw1); (5,8,0,0:vw1); (5,9,0,0:vw1); (5,10,0,0:vw1); (5,11,0,0:vw1); (5,12,0,0:vw1); (5,13,0,0:vw1); (5,14,0,0:vw1); (5,15,0,0:vw1); (6,0,0,0:vw1); (6,1,0,0:vw1); (6,2,0,0:vw1); (6,3,0,0:vw1); (6,4,0,0:vw1); (6,5,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v134, v130, v128, 1   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_short_d16 v136, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_short_d16 v137, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_short_d16 v138, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v139, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
buffer_load_short_d16 v140, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
buffer_load_short_d16 v141, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
buffer_load_short_d16 v142, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,8,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v143, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,9,0) */
buffer_load_short_d16 v144, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,10,0) */
buffer_load_short_d16 v145, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,11,0) */
buffer_load_short_d16 v146, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,12,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v147, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,13,0) */
buffer_load_short_d16 v148, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,14,0) */
buffer_load_short_d16 v149, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(0,0,15,0) */
buffer_load_short_d16 v150, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v151, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_short_d16 v152, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
buffer_load_short_d16 v153, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
buffer_load_short_d16 v154, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v155, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
buffer_load_short_d16 v156, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
buffer_load_short_d16 v157, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
buffer_load_short_d16 v158, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,8,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v159, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,9,0) */
buffer_load_short_d16 v160, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,10,0) */
buffer_load_short_d16 v161, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,11,0) */
buffer_load_short_d16 v162, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,12,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v163, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,13,0) */
buffer_load_short_d16 v164, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,14,0) */
buffer_load_short_d16 v165, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(1,0,15,0) */
buffer_load_short_d16 v166, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v167, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_short_d16 v168, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
buffer_load_short_d16 v169, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
buffer_load_short_d16 v170, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v171, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,5,0) */
buffer_load_short_d16 v172, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,6,0) */
buffer_load_short_d16 v173, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,7,0) */
buffer_load_short_d16 v174, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,8,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v175, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,9,0) */
buffer_load_short_d16 v176, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,10,0) */
buffer_load_short_d16 v177, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,11,0) */
buffer_load_short_d16 v178, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,12,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v179, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,13,0) */
buffer_load_short_d16 v180, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,14,0) */
buffer_load_short_d16 v181, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(2,0,15,0) */
buffer_load_short_d16 v182, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v183, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_short_d16 v184, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
buffer_load_short_d16 v185, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
buffer_load_short_d16 v186, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v187, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,5,0) */
buffer_load_short_d16 v188, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,6,0) */
buffer_load_short_d16 v189, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,7,0) */
buffer_load_short_d16 v190, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,8,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v191, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,9,0) */
buffer_load_short_d16 v192, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,10,0) */
buffer_load_short_d16 v193, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,11,0) */
buffer_load_short_d16 v194, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,12,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v195, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,13,0) */
buffer_load_short_d16 v196, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,14,0) */
buffer_load_short_d16 v197, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(3,0,15,0) */
buffer_load_short_d16 v201, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
_v_add_lshl_u32 v134, v130, v128, 1 
s_mul_i32 s54, s[sgprStrideC1J], 32     // scale StrideC *= numRows(16) * bpe
s_add_u32  s[sgprSrdC+0], s[sgprSrdC+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdC+1], s[sgprSrdC+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_load_short_d16 v202, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_short_d16 v203, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,2,0) */
buffer_load_short_d16 v204, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,3,0) */
buffer_load_short_d16 v205, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,4,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v206, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,5,0) */
buffer_load_short_d16 v207, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,6,0) */
buffer_load_short_d16 v208, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,7,0) */
buffer_load_short_d16 v209, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,8,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v210, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,9,0) */
buffer_load_short_d16 v211, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,10,0) */
buffer_load_short_d16 v212, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,11,0) */
buffer_load_short_d16 v213, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,12,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v214, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,13,0) */
buffer_load_short_d16 v215, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,14,0) */
buffer_load_short_d16 v216, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(4,0,15,0) */
buffer_load_short_d16 v217, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,0,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v218, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,1,0) */
buffer_load_short_d16 v219, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,2,0) */
buffer_load_short_d16 v220, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,3,0) */
buffer_load_short_d16 v221, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc


/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 8, 0, 0), (0, 9, 0, 0), (0, 10, 0, 0), (0, 11, 0, 0), (0, 12, 0, 0), (0, 13, 0, 0), (0, 14, 0, 0), (0, 15, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0), (1, 8, 0, 0), (1, 9, 0, 0), (1, 10, 0, 0), (1, 11, 0, 0), (1, 12, 0, 0), (1, 13, 0, 0), (1, 14, 0, 0), (1, 15, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (2, 5, 0, 0), (2, 6, 0, 0), (2, 7, 0, 0), (2, 8, 0, 0), (2, 9, 0, 0), (2, 10, 0, 0), (2, 11, 0, 0), (2, 12, 0, 0), (2, 13, 0, 0), (2, 14, 0, 0), (2, 15, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (3, 5, 0, 0), (3, 6, 0, 0), (3, 7, 0, 0), (3, 8, 0, 0), (3, 9, 0, 0), (3, 10, 0, 0), (3, 11, 0, 0), (3, 12, 0, 0), (3, 13, 0, 0), (3, 14, 0, 0), (3, 15, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0), (4, 2, 0, 0), (4, 3, 0, 0), (4, 4, 0, 0), (4, 5, 0, 0), (4, 6, 0, 0), (4, 7, 0, 0), (4, 8, 0, 0), (4, 9, 0, 0), (4, 10, 0, 0), (4, 11, 0, 0), (4, 12, 0, 0), (4, 13, 0, 0), (4, 14, 0, 0), (4, 15, 0, 0), (5, 0, 0, 0), (5, 1, 0, 0), (5, 2, 0, 0), (5, 3, 0, 0), (5, 4, 0, 0), (5, 5, 0, 0), (5, 6, 0, 0), (5, 7, 0, 0), (5, 8, 0, 0), (5, 9, 0, 0), (5, 10, 0, 0), (5, 11, 0, 0), (5, 12, 0, 0), (5, 13, 0, 0), (5, 14, 0, 0), (5, 15, 0, 0), (6, 0, 0, 0), (6, 1, 0, 0), (6, 2, 0, 0), (6, 3, 0, 0), (6, 4, 0, 0), (6, 5, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha

/* apply mask, calc new C and issue writes */

_v_add_lshl_u32 v134, v131, v128, 1   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0
s_waitcnt vmcnt(0)                                // wait C (interleaved) 101 = 102 - 0 + 0 - 1
BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 136
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 138
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 140
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 142
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 144
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 146
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 148
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 150
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 152
BF16_TO_FP32_Single 153
BF16_TO_FP32_Single 154
BF16_TO_FP32_Single 155
BF16_TO_FP32_Single 156
BF16_TO_FP32_Single 157
BF16_TO_FP32_Single 158
BF16_TO_FP32_Single 159
BF16_TO_FP32_Single 160
BF16_TO_FP32_Single 161
BF16_TO_FP32_Single 162
BF16_TO_FP32_Single 163
BF16_TO_FP32_Single 164
BF16_TO_FP32_Single 165
BF16_TO_FP32_Single 166
BF16_TO_FP32_Single 167
BF16_TO_FP32_Single 168
BF16_TO_FP32_Single 169
BF16_TO_FP32_Single 170
BF16_TO_FP32_Single 171
BF16_TO_FP32_Single 172
BF16_TO_FP32_Single 173
BF16_TO_FP32_Single 174
BF16_TO_FP32_Single 175
BF16_TO_FP32_Single 176
BF16_TO_FP32_Single 177
BF16_TO_FP32_Single 178
BF16_TO_FP32_Single 179
BF16_TO_FP32_Single 180
BF16_TO_FP32_Single 181
BF16_TO_FP32_Single 182
BF16_TO_FP32_Single 183
BF16_TO_FP32_Single 184
BF16_TO_FP32_Single 185
BF16_TO_FP32_Single 186
BF16_TO_FP32_Single 187
BF16_TO_FP32_Single 188
BF16_TO_FP32_Single 189
BF16_TO_FP32_Single 190
BF16_TO_FP32_Single 191
BF16_TO_FP32_Single 192
BF16_TO_FP32_Single 193
BF16_TO_FP32_Single 194
BF16_TO_FP32_Single 195
BF16_TO_FP32_Single 196
BF16_TO_FP32_Single 197
BF16_TO_FP32_Single 201
BF16_TO_FP32_Single 202
BF16_TO_FP32_Single 203
BF16_TO_FP32_Single 204
BF16_TO_FP32_Single 205
BF16_TO_FP32_Single 206
BF16_TO_FP32_Single 207
BF16_TO_FP32_Single 208
BF16_TO_FP32_Single 209
BF16_TO_FP32_Single 210
BF16_TO_FP32_Single 211
BF16_TO_FP32_Single 212
BF16_TO_FP32_Single 213
BF16_TO_FP32_Single 214
BF16_TO_FP32_Single 215
BF16_TO_FP32_Single 216
BF16_TO_FP32_Single 217
BF16_TO_FP32_Single 218
BF16_TO_FP32_Single 219
BF16_TO_FP32_Single 220
BF16_TO_FP32_Single 221
v_mac_f32 v[vgprValuC+0], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+1], v136, s[sgprBeta]
v_mac_f32 v[vgprValuC+2], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+3], v138, s[sgprBeta]
v_mac_f32 v[vgprValuC+4], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+5], v140, s[sgprBeta]
v_mac_f32 v[vgprValuC+6], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+7], v142, s[sgprBeta]
v_mac_f32 v[vgprValuC+8], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+9], v144, s[sgprBeta]
v_mac_f32 v[vgprValuC+10], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+11], v146, s[sgprBeta]
v_mac_f32 v[vgprValuC+12], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+13], v148, s[sgprBeta]
v_mac_f32 v[vgprValuC+14], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+15], v150, s[sgprBeta]
v_mac_f32 v[vgprValuC+16], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+17], v152, s[sgprBeta]
v_mac_f32 v[vgprValuC+18], v153, s[sgprBeta]
v_mac_f32 v[vgprValuC+19], v154, s[sgprBeta]
v_mac_f32 v[vgprValuC+20], v155, s[sgprBeta]
v_mac_f32 v[vgprValuC+21], v156, s[sgprBeta]
v_mac_f32 v[vgprValuC+22], v157, s[sgprBeta]
v_mac_f32 v[vgprValuC+23], v158, s[sgprBeta]
v_mac_f32 v[vgprValuC+24], v159, s[sgprBeta]
v_mac_f32 v[vgprValuC+25], v160, s[sgprBeta]
v_mac_f32 v[vgprValuC+26], v161, s[sgprBeta]
v_mac_f32 v[vgprValuC+27], v162, s[sgprBeta]
v_mac_f32 v[vgprValuC+28], v163, s[sgprBeta]
v_mac_f32 v[vgprValuC+29], v164, s[sgprBeta]
v_mac_f32 v[vgprValuC+30], v165, s[sgprBeta]
v_mac_f32 v[vgprValuC+31], v166, s[sgprBeta]
v_mac_f32 v[vgprValuC+32], v167, s[sgprBeta]
v_mac_f32 v[vgprValuC+33], v168, s[sgprBeta]
v_mac_f32 v[vgprValuC+34], v169, s[sgprBeta]
v_mac_f32 v[vgprValuC+35], v170, s[sgprBeta]
v_mac_f32 v[vgprValuC+36], v171, s[sgprBeta]
v_mac_f32 v[vgprValuC+37], v172, s[sgprBeta]
v_mac_f32 v[vgprValuC+38], v173, s[sgprBeta]
v_mac_f32 v[vgprValuC+39], v174, s[sgprBeta]
v_mac_f32 v[vgprValuC+40], v175, s[sgprBeta]
v_mac_f32 v[vgprValuC+41], v176, s[sgprBeta]
v_mac_f32 v[vgprValuC+42], v177, s[sgprBeta]
v_mac_f32 v[vgprValuC+43], v178, s[sgprBeta]
v_mac_f32 v[vgprValuC+44], v179, s[sgprBeta]
v_mac_f32 v[vgprValuC+45], v180, s[sgprBeta]
v_mac_f32 v[vgprValuC+46], v181, s[sgprBeta]
v_mac_f32 v[vgprValuC+47], v182, s[sgprBeta]
v_mac_f32 v[vgprValuC+48], v183, s[sgprBeta]
v_mac_f32 v[vgprValuC+49], v184, s[sgprBeta]
v_mac_f32 v[vgprValuC+50], v185, s[sgprBeta]
v_mac_f32 v[vgprValuC+51], v186, s[sgprBeta]
v_mac_f32 v[vgprValuC+52], v187, s[sgprBeta]
v_mac_f32 v[vgprValuC+53], v188, s[sgprBeta]
v_mac_f32 v[vgprValuC+54], v189, s[sgprBeta]
v_mac_f32 v[vgprValuC+55], v190, s[sgprBeta]
v_mac_f32 v[vgprValuC+56], v191, s[sgprBeta]
v_mac_f32 v[vgprValuC+57], v192, s[sgprBeta]
v_mac_f32 v[vgprValuC+58], v193, s[sgprBeta]
v_mac_f32 v[vgprValuC+59], v194, s[sgprBeta]
v_mac_f32 v[vgprValuC+60], v195, s[sgprBeta]
v_mac_f32 v[vgprValuC+61], v196, s[sgprBeta]
v_mac_f32 v[vgprValuC+62], v197, s[sgprBeta]
v_mac_f32 v[vgprValuC+63], v201, s[sgprBeta]
v_mac_f32 v[vgprValuC+64], v202, s[sgprBeta]
v_mac_f32 v[vgprValuC+65], v203, s[sgprBeta]
v_mac_f32 v[vgprValuC+66], v204, s[sgprBeta]
v_mac_f32 v[vgprValuC+67], v205, s[sgprBeta]
v_mac_f32 v[vgprValuC+68], v206, s[sgprBeta]
v_mac_f32 v[vgprValuC+69], v207, s[sgprBeta]
v_mac_f32 v[vgprValuC+70], v208, s[sgprBeta]
v_mac_f32 v[vgprValuC+71], v209, s[sgprBeta]
v_mac_f32 v[vgprValuC+72], v210, s[sgprBeta]
v_mac_f32 v[vgprValuC+73], v211, s[sgprBeta]
v_mac_f32 v[vgprValuC+74], v212, s[sgprBeta]
v_mac_f32 v[vgprValuC+75], v213, s[sgprBeta]
v_mac_f32 v[vgprValuC+76], v214, s[sgprBeta]
v_mac_f32 v[vgprValuC+77], v215, s[sgprBeta]
v_mac_f32 v[vgprValuC+78], v216, s[sgprBeta]
v_mac_f32 v[vgprValuC+79], v217, s[sgprBeta]
v_mac_f32 v[vgprValuC+80], v218, s[sgprBeta]
v_mac_f32 v[vgprValuC+81], v219, s[sgprBeta]
v_mac_f32 v[vgprValuC+82], v220, s[sgprBeta]
v_mac_f32 v[vgprValuC+83], v221, s[sgprBeta]

label_BiasAddrValid_End_2_1:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 0
FP32_TO_BF16_Single 1
FP32_TO_BF16_Single 2
FP32_TO_BF16_Single 3
FP32_TO_BF16_Single 4
FP32_TO_BF16_Single 5
FP32_TO_BF16_Single 6
FP32_TO_BF16_Single 7
FP32_TO_BF16_Single 8
FP32_TO_BF16_Single 9
FP32_TO_BF16_Single 10
FP32_TO_BF16_Single 11
FP32_TO_BF16_Single 12
FP32_TO_BF16_Single 13
FP32_TO_BF16_Single 14
FP32_TO_BF16_Single 15
FP32_TO_BF16_Single 16
FP32_TO_BF16_Single 17
FP32_TO_BF16_Single 18
FP32_TO_BF16_Single 19
FP32_TO_BF16_Single 20
FP32_TO_BF16_Single 21
FP32_TO_BF16_Single 22
FP32_TO_BF16_Single 23
FP32_TO_BF16_Single 24
FP32_TO_BF16_Single 25
FP32_TO_BF16_Single 26
FP32_TO_BF16_Single 27
FP32_TO_BF16_Single 28
FP32_TO_BF16_Single 29
FP32_TO_BF16_Single 30
FP32_TO_BF16_Single 31
FP32_TO_BF16_Single 32
FP32_TO_BF16_Single 33
FP32_TO_BF16_Single 34
FP32_TO_BF16_Single 35
FP32_TO_BF16_Single 36
FP32_TO_BF16_Single 37
FP32_TO_BF16_Single 38
FP32_TO_BF16_Single 39
FP32_TO_BF16_Single 40
FP32_TO_BF16_Single 41
FP32_TO_BF16_Single 42
FP32_TO_BF16_Single 43
FP32_TO_BF16_Single 44
FP32_TO_BF16_Single 45
FP32_TO_BF16_Single 46
FP32_TO_BF16_Single 47
FP32_TO_BF16_Single 48
FP32_TO_BF16_Single 49
FP32_TO_BF16_Single 50
FP32_TO_BF16_Single 51
FP32_TO_BF16_Single 52
FP32_TO_BF16_Single 53
FP32_TO_BF16_Single 54
FP32_TO_BF16_Single 55
FP32_TO_BF16_Single 56
FP32_TO_BF16_Single 57
FP32_TO_BF16_Single 58
FP32_TO_BF16_Single 59
FP32_TO_BF16_Single 60
FP32_TO_BF16_Single 61
FP32_TO_BF16_Single 62
FP32_TO_BF16_Single 63
FP32_TO_BF16_Single 64
FP32_TO_BF16_Single 65
FP32_TO_BF16_Single 66
FP32_TO_BF16_Single 67
FP32_TO_BF16_Single 68
FP32_TO_BF16_Single 69
FP32_TO_BF16_Single 70
FP32_TO_BF16_Single 71
FP32_TO_BF16_Single 72
FP32_TO_BF16_Single 73
FP32_TO_BF16_Single 74
FP32_TO_BF16_Single 75
FP32_TO_BF16_Single 76
FP32_TO_BF16_Single 77
FP32_TO_BF16_Single 78
FP32_TO_BF16_Single 79
FP32_TO_BF16_Single 80
FP32_TO_BF16_Single 81
FP32_TO_BF16_Single 82
FP32_TO_BF16_Single 83
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v1, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v2, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v3, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v4, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v5, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v6, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v7, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v8, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v9, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v10, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v11, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v12, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v13, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v14, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v15, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v16, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v17, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v18, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v19, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v21, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v22, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v23, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v24, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v25, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v26, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v27, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v28, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v29, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v30, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v31, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v32, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v33, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v34, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v35, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v36, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v37, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v38, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v39, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v41, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v42, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v43, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v44, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v45, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v46, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v47, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v48, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v49, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v50, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v51, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v52, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v53, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v54, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v55, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v56, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v57, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v58, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v59, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v61, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v62, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v63, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_lshl_u32 v134, v131, v128, 1 
s_mul_i32 s54, s[sgprStrideD1J], 32     // scale StrideD *= numRows(16) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s54       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
buffer_store_short v64, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v65, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v66, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v67, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v68, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v69, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v70, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v71, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v72, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v73, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v74, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v75, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v76, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v77, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v78, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v79, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v81, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v82, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v83, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D


/* optSingleColVgpr=1 optSharedColVgpr=0 optSharedMask=1 optSrdIncForRow=0 */

/* Global Write Beta Batch #1-1 */

/* (d1,vc1,d0,vc0)=(5,0,4,0) */
_v_add_lshl_u32 v134, v130, v128, 1   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0
s_mov_b32 s54, 160
v_add_co_u32 v134, vcc, s54, v134                 // add offset 
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,5,0) */
buffer_load_short_d16 v136, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,6,0) */
buffer_load_short_d16 v137, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,7,0) */
buffer_load_short_d16 v138, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,8,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v139, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,9,0) */
buffer_load_short_d16 v140, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,10,0) */
buffer_load_short_d16 v141, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,11,0) */
buffer_load_short_d16 v142, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,12,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v143, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,13,0) */
buffer_load_short_d16 v144, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,14,0) */
buffer_load_short_d16 v145, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(5,0,15,0) */
buffer_load_short_d16 v146, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,0,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v147, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,1,0) */
buffer_load_short_d16 v148, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,2,0) */
buffer_load_short_d16 v149, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,3,0) */
buffer_load_short_d16 v150, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,4,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v151, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,5,0) */
buffer_load_short_d16 v152, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc

v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha

s_waitcnt vmcnt(0)  

_v_add_lshl_u32 v134, v131, v128, 1   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0
s_mov_b32 s54, 160
v_add_co_u32 v134, vcc, s54, v134                 // add offset

BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 136
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 138
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 140
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 142
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 144
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 146
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 148
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 150
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 152
v_mac_f32 v[vgprValuC+84], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+85], v136, s[sgprBeta]
v_mac_f32 v[vgprValuC+86], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+87], v138, s[sgprBeta]
v_mac_f32 v[vgprValuC+88], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+89], v140, s[sgprBeta]
v_mac_f32 v[vgprValuC+90], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+91], v142, s[sgprBeta]
v_mac_f32 v[vgprValuC+92], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+93], v144, s[sgprBeta]
v_mac_f32 v[vgprValuC+94], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+95], v146, s[sgprBeta]
v_mac_f32 v[vgprValuC+96], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+97], v148, s[sgprBeta]
v_mac_f32 v[vgprValuC+98], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+99], v150, s[sgprBeta]
v_mac_f32 v[vgprValuC+100], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+101], v152, s[sgprBeta]

label_BiasAddrValid_End_2_2:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 84
FP32_TO_BF16_Single 85
FP32_TO_BF16_Single 86
FP32_TO_BF16_Single 87
FP32_TO_BF16_Single 88
FP32_TO_BF16_Single 89
FP32_TO_BF16_Single 90
FP32_TO_BF16_Single 91
FP32_TO_BF16_Single 92
FP32_TO_BF16_Single 93
FP32_TO_BF16_Single 94
FP32_TO_BF16_Single 95
FP32_TO_BF16_Single 96
FP32_TO_BF16_Single 97
FP32_TO_BF16_Single 98
FP32_TO_BF16_Single 99
FP32_TO_BF16_Single 100
FP32_TO_BF16_Single 101
buffer_store_short v84, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v85, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v86, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v87, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v88, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v89, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v90, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v91, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v92, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v93, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v94, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v95, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v96, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v97, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v98, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v99, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20 
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v101, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D

/* Global Write Beta Batch #1-1 */


/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (6,6,0,0:vw1); (6,7,0,0:vw1); (6,8,0,0:vw1); (6,9,0,0:vw1); (6,10,0,0:vw1); (6,11,0,0:vw1); (6,12,0,0:vw1); (6,13,0,0:vw1); (6,14,0,0:vw1); (6,15,0,0:vw1); (7,0,0,0:vw1); (7,1,0,0:vw1); (7,2,0,0:vw1); (7,3,0,0:vw1); (7,4,0,0:vw1); (7,5,0,0:vw1); (7,6,0,0:vw1); (7,7,0,0:vw1); (7,8,0,0:vw1); (7,9,0,0:vw1); (7,10,0,0:vw1); (7,11,0,0:vw1); (7,12,0,0:vw1); (7,13,0,0:vw1); (7,14,0,0:vw1); (7,15,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(6,0,6,0) */
_v_add_lshl_u32 v134, v130, v128, 1   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0
s_mov_b32 s54, 288
v_add_co_u32 v134, vcc, s54, v134                 // add offset 
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,7,0) */
buffer_load_short_d16 v136, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,8,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v137, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,9,0) */
buffer_load_short_d16 v138, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,10,0) */
buffer_load_short_d16 v139, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,11,0) */
buffer_load_short_d16 v140, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,12,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v141, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,13,0) */
buffer_load_short_d16 v142, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,14,0) */
buffer_load_short_d16 v143, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(6,0,15,0) */
buffer_load_short_d16 v144, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,0,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v145, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,1,0) */
buffer_load_short_d16 v146, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,2,0) */
buffer_load_short_d16 v147, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,3,0) */
buffer_load_short_d16 v148, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,4,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v149, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,5,0) */
buffer_load_short_d16 v150, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,6,0) */
buffer_load_short_d16 v151, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,7,0) */
buffer_load_short_d16 v152, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,8,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v153, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,9,0) */
buffer_load_short_d16 v154, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,10,0) */
buffer_load_short_d16 v155, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,11,0) */
buffer_load_short_d16 v156, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,12,0) */
_v_add_co_u32 v134, vcc, v134, 32   // load not edge offset
buffer_load_short_d16 v157, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,13,0) */
buffer_load_short_d16 v158, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+0], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,14,0) */
buffer_load_short_d16 v159, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+1], offen, offset:0 // load C for beta calc
/* (d1,vc1,d0,vc0)=(7,0,15,0) */
buffer_load_short_d16 v160, v134, s[sgprSrdC:sgprSrdC+3], s[sgprStrideoffC+2], offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 1   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0
s_mov_b32 s54, 288
v_add_co_u32 v134, vcc, s54, v134                 // add offset

/* rC *= alpha batchEements=[(6, 6, 0, 0), (6, 7, 0, 0), (6, 8, 0, 0), (6, 9, 0, 0), (6, 10, 0, 0), (6, 11, 0, 0), (6, 12, 0, 0), (6, 13, 0, 0), (6, 14, 0, 0), (6, 15, 0, 0), (7, 0, 0, 0), (7, 1, 0, 0), (7, 2, 0, 0), (7, 3, 0, 0), (7, 4, 0, 0), (7, 5, 0, 0), (7, 6, 0, 0), (7, 7, 0, 0), (7, 8, 0, 0), (7, 9, 0, 0), (7, 10, 0, 0), (7, 11, 0, 0), (7, 12, 0, 0), (7, 13, 0, 0), (7, 14, 0, 0), (7, 15, 0, 0)] */
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(0)                                // wait C (interleaved) 25 = 26 - 0 + 0 - 1

BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 136
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 138
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 140
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 142
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 144
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 146
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 148
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 150
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 152
BF16_TO_FP32_Single 153
BF16_TO_FP32_Single 154
BF16_TO_FP32_Single 155
BF16_TO_FP32_Single 156
BF16_TO_FP32_Single 157
BF16_TO_FP32_Single 158
BF16_TO_FP32_Single 159
BF16_TO_FP32_Single 160
v_mac_f32 v[vgprValuC+102], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+103], v136, s[sgprBeta]
v_mac_f32 v[vgprValuC+104], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+105], v138, s[sgprBeta]
v_mac_f32 v[vgprValuC+106], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+107], v140, s[sgprBeta]
v_mac_f32 v[vgprValuC+108], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+109], v142, s[sgprBeta]
v_mac_f32 v[vgprValuC+110], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+111], v144, s[sgprBeta]
v_mac_f32 v[vgprValuC+112], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+113], v146, s[sgprBeta]
v_mac_f32 v[vgprValuC+114], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+115], v148, s[sgprBeta]
v_mac_f32 v[vgprValuC+116], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+117], v150, s[sgprBeta]
v_mac_f32 v[vgprValuC+118], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+119], v152, s[sgprBeta]
v_mac_f32 v[vgprValuC+120], v153, s[sgprBeta]
v_mac_f32 v[vgprValuC+121], v154, s[sgprBeta]
v_mac_f32 v[vgprValuC+122], v155, s[sgprBeta]
v_mac_f32 v[vgprValuC+123], v156, s[sgprBeta]
v_mac_f32 v[vgprValuC+124], v157, s[sgprBeta]
v_mac_f32 v[vgprValuC+125], v158, s[sgprBeta]
v_mac_f32 v[vgprValuC+126], v159, s[sgprBeta]
v_mac_f32 v[vgprValuC+127], v160, s[sgprBeta]

label_BiasAddrValid_End_2_3:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 102
FP32_TO_BF16_Single 103
FP32_TO_BF16_Single 104
FP32_TO_BF16_Single 105
FP32_TO_BF16_Single 106
FP32_TO_BF16_Single 107
FP32_TO_BF16_Single 108
FP32_TO_BF16_Single 109
FP32_TO_BF16_Single 110
FP32_TO_BF16_Single 111
FP32_TO_BF16_Single 112
FP32_TO_BF16_Single 113
FP32_TO_BF16_Single 114
FP32_TO_BF16_Single 115
FP32_TO_BF16_Single 116
FP32_TO_BF16_Single 117
FP32_TO_BF16_Single 118
FP32_TO_BF16_Single 119
FP32_TO_BF16_Single 120
FP32_TO_BF16_Single 121
FP32_TO_BF16_Single 122
FP32_TO_BF16_Single 123
FP32_TO_BF16_Single 124
FP32_TO_BF16_Single 125
FP32_TO_BF16_Single 126
FP32_TO_BF16_Single 127
buffer_store_short v102, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v103, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20
buffer_store_short v104, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v105, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v106, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v107, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20
buffer_store_short v108, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v109, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v110, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v111, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20
buffer_store_short v112, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v113, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v114, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v115, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20
buffer_store_short v116, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v117, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v118, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v119, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v121, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v122, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v123, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D
_v_add_co_u32 v134, vcc, v134, 0x20
buffer_store_short v124, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v125, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+0], offen, offset:0 // store D
buffer_store_short v126, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+1], offen, offset:0 // store D
buffer_store_short v127, v134, s[sgprSrdD:sgprSrdD+3], s[sgprStrideoffD+2], offen, offset:0 // store D

s_branch label_GW_End_37                           // jump to end
GW_B1_E1_36:

/* edge=1, allocate 48 sgpr. perBatch=6 perElement=2 elementsPerBatch=21 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */
/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,1,0,4:vw1); (0,1,0,5:vw1); (0,1,0,6:vw1); (0,1,0,7:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,4) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,5) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,6) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,7) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 1, 0, 4), (0, 1, 0, 5), (0, 1, 0, 6), (0, 1, 0, 7), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 153
BF16_TO_FP32_Single 155
BF16_TO_FP32_Single 157
BF16_TO_FP32_Single 159
BF16_TO_FP32_Single 161
BF16_TO_FP32_Single 163
BF16_TO_FP32_Single 165
BF16_TO_FP32_Single 167
BF16_TO_FP32_Single 169
BF16_TO_FP32_Single 171
BF16_TO_FP32_Single 173
v_mac_f32 v[vgprValuC+0], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+1], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+2], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+3], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+4], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+5], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+6], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+7], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+8], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+9], v153, s[sgprBeta]
v_mac_f32 v[vgprValuC+10], v155, s[sgprBeta]
v_mac_f32 v[vgprValuC+11], v157, s[sgprBeta]
v_mac_f32 v[vgprValuC+12], v159, s[sgprBeta]
v_mac_f32 v[vgprValuC+13], v161, s[sgprBeta]
v_mac_f32 v[vgprValuC+14], v163, s[sgprBeta]
v_mac_f32 v[vgprValuC+15], v165, s[sgprBeta]
v_mac_f32 v[vgprValuC+16], v167, s[sgprBeta]
v_mac_f32 v[vgprValuC+17], v169, s[sgprBeta]
v_mac_f32 v[vgprValuC+18], v171, s[sgprBeta]
v_mac_f32 v[vgprValuC+19], v173, s[sgprBeta]

label_BiasAddrValid_End_2_4:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 0
FP32_TO_BF16_Single 1
FP32_TO_BF16_Single 2
FP32_TO_BF16_Single 3
FP32_TO_BF16_Single 4
FP32_TO_BF16_Single 5
FP32_TO_BF16_Single 6
FP32_TO_BF16_Single 7
FP32_TO_BF16_Single 8
FP32_TO_BF16_Single 9
FP32_TO_BF16_Single 10
FP32_TO_BF16_Single 11
FP32_TO_BF16_Single 12
FP32_TO_BF16_Single 13
FP32_TO_BF16_Single 14
FP32_TO_BF16_Single 15
FP32_TO_BF16_Single 16
FP32_TO_BF16_Single 17
FP32_TO_BF16_Single 18
FP32_TO_BF16_Single 19
buffer_store_short v0, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v1, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v2, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v3, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v4, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v5, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v6, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v7, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v8, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v9, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v10, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v11, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v12, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v13, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v14, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v15, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v16, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v17, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v18, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v19, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,1,4:vw1); (0,0,1,5:vw1); (0,0,1,6:vw1); (0,0,1,7:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,1,1,4:vw1); (0,1,1,5:vw1); (0,1,1,6:vw1); (0,1,1,7:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,2,4:vw1); (0,0,2,5:vw1); (0,0,2,6:vw1); (0,0,2,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,1,0,4) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,5) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,6) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,7) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,4) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v129, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,5) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 4                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,6) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,7) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset

/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,4) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,7) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 1, 4), (0, 0, 1, 5), (0, 0, 1, 6), (0, 0, 1, 7), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 1, 1, 4), (0, 1, 1, 5), (0, 1, 1, 6), (0, 1, 1, 7), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 2, 4), (0, 0, 2, 5), (0, 0, 2, 6), (0, 0, 2, 7)] */
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 153
BF16_TO_FP32_Single 155
BF16_TO_FP32_Single 157
BF16_TO_FP32_Single 159
BF16_TO_FP32_Single 161
BF16_TO_FP32_Single 163
BF16_TO_FP32_Single 165
BF16_TO_FP32_Single 167
BF16_TO_FP32_Single 169
BF16_TO_FP32_Single 171
BF16_TO_FP32_Single 173
v_mac_f32 v[vgprValuC+20], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+21], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+22], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+23], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+24], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+25], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+26], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+27], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+28], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+29], v153, s[sgprBeta]
v_mac_f32 v[vgprValuC+30], v155, s[sgprBeta]
v_mac_f32 v[vgprValuC+31], v157, s[sgprBeta]
v_mac_f32 v[vgprValuC+32], v159, s[sgprBeta]
v_mac_f32 v[vgprValuC+33], v161, s[sgprBeta]
v_mac_f32 v[vgprValuC+34], v163, s[sgprBeta]
v_mac_f32 v[vgprValuC+35], v165, s[sgprBeta]
v_mac_f32 v[vgprValuC+36], v167, s[sgprBeta]
v_mac_f32 v[vgprValuC+37], v169, s[sgprBeta]
v_mac_f32 v[vgprValuC+38], v171, s[sgprBeta]
v_mac_f32 v[vgprValuC+39], v173, s[sgprBeta]

label_BiasAddrValid_End_2_5:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 20
FP32_TO_BF16_Single 21
FP32_TO_BF16_Single 22
FP32_TO_BF16_Single 23
FP32_TO_BF16_Single 24
FP32_TO_BF16_Single 25
FP32_TO_BF16_Single 26
FP32_TO_BF16_Single 27
FP32_TO_BF16_Single 28
FP32_TO_BF16_Single 29
FP32_TO_BF16_Single 30
FP32_TO_BF16_Single 31
FP32_TO_BF16_Single 32
FP32_TO_BF16_Single 33
FP32_TO_BF16_Single 34
FP32_TO_BF16_Single 35
FP32_TO_BF16_Single 36
FP32_TO_BF16_Single 37
FP32_TO_BF16_Single 38
FP32_TO_BF16_Single 39
buffer_store_short v20, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v21, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v22, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v23, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v24, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v25, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v26, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v27, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v28, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v29, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v30, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v31, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v32, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v33, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v34, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v35, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v36, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v37, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v38, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v39, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,1,2,4:vw1); (0,1,2,5:vw1); (0,1,2,6:vw1); (0,1,2,7:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,3,4:vw1); (0,0,3,5:vw1); (0,0,3,6:vw1); (0,0,3,7:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,4) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,7) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,4) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,7) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (0, 1, 2, 7), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 3, 4), (0, 0, 3, 5), (0, 0, 3, 6), (0, 0, 3, 7), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3)] */
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 153
BF16_TO_FP32_Single 155
BF16_TO_FP32_Single 157
BF16_TO_FP32_Single 159
BF16_TO_FP32_Single 161
BF16_TO_FP32_Single 163
BF16_TO_FP32_Single 165
BF16_TO_FP32_Single 167
BF16_TO_FP32_Single 169
BF16_TO_FP32_Single 171
BF16_TO_FP32_Single 173
v_mac_f32 v[vgprValuC+40], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+41], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+42], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+43], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+44], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+45], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+46], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+47], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+48], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+49], v153, s[sgprBeta]
v_mac_f32 v[vgprValuC+50], v155, s[sgprBeta]
v_mac_f32 v[vgprValuC+51], v157, s[sgprBeta]
v_mac_f32 v[vgprValuC+52], v159, s[sgprBeta]
v_mac_f32 v[vgprValuC+53], v161, s[sgprBeta]
v_mac_f32 v[vgprValuC+54], v163, s[sgprBeta]
v_mac_f32 v[vgprValuC+55], v165, s[sgprBeta]
v_mac_f32 v[vgprValuC+56], v167, s[sgprBeta]
v_mac_f32 v[vgprValuC+57], v169, s[sgprBeta]
v_mac_f32 v[vgprValuC+58], v171, s[sgprBeta]
v_mac_f32 v[vgprValuC+59], v173, s[sgprBeta]

label_BiasAddrValid_End_2_6:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 40
FP32_TO_BF16_Single 41
FP32_TO_BF16_Single 42
FP32_TO_BF16_Single 43
FP32_TO_BF16_Single 44
FP32_TO_BF16_Single 45
FP32_TO_BF16_Single 46
FP32_TO_BF16_Single 47
FP32_TO_BF16_Single 48
FP32_TO_BF16_Single 49
FP32_TO_BF16_Single 50
FP32_TO_BF16_Single 51
FP32_TO_BF16_Single 52
FP32_TO_BF16_Single 53
FP32_TO_BF16_Single 54
FP32_TO_BF16_Single 55
FP32_TO_BF16_Single 56
FP32_TO_BF16_Single 57
FP32_TO_BF16_Single 58
FP32_TO_BF16_Single 59
buffer_store_short v40, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v41, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v42, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v43, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v44, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v45, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v46, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v47, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v48, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v49, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v50, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v51, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v52, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v53, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v54, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v55, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v56, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v57, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v58, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v59, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,1,3,4:vw1); (0,1,3,5:vw1); (0,1,3,6:vw1); (0,1,3,7:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,4,4:vw1); (0,0,4,5:vw1); (0,0,4,6:vw1); (0,0,4,7:vw1); (0,1,4,0:vw1); (0,1,4,1:vw1); (0,1,4,2:vw1); (0,1,4,3:vw1); (0,1,4,4:vw1); (0,1,4,5:vw1); (0,1,4,6:vw1); (0,1,4,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,3,1,4) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 0                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,5) */
_v_add_co_u32 v133, vcc, v129, 4                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,6) */
_v_add_co_u32 v133, vcc, v129, 8                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,7) */
_v_add_co_u32 v133, vcc, v129, 12                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset


/* (d1,vc1,d0,vc0)=(0,4,0,0) */
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck], s[sgprStrideC1J], 32  // add offset stride

v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
s_mul_i32 s[sgprEdgeCheck+1], s[sgprStrideD1J], 32  // add offset stride

v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v128, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v128, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,4) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck], v150     // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,5) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck], v152     // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,6) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck], v154     // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,7) */
_v_add_co_u32 v132, vcc, v128, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck], v156     // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck+1], v156   // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,0) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck], v158     // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck+1], v158   // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,1) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck], v160     // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck+1], v160   // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,2) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck], v162     // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck+1], v162   // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,3) */
_v_add_co_u32 v132, vcc, v128, 32                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck], v164     // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck+1], v164   // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,4) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck], v166     // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck+1], v166   // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,5) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck], v168     // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck+1], v168   // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,6) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck], v170     // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck+1], v170   // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,1,7) */
_v_add_co_u32 v132, vcc, v128, 48                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck], v172     // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck+1], v172   // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 3, 7), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 4, 4), (0, 0, 4, 5), (0, 0, 4, 6), (0, 0, 4, 7), (0, 1, 4, 0), (0, 1, 4, 1), (0, 1, 4, 2), (0, 1, 4, 3), (0, 1, 4, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 1, 4, 7)] */
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 153
BF16_TO_FP32_Single 155
BF16_TO_FP32_Single 157
BF16_TO_FP32_Single 159
BF16_TO_FP32_Single 161
BF16_TO_FP32_Single 163
BF16_TO_FP32_Single 165
BF16_TO_FP32_Single 167
BF16_TO_FP32_Single 169
BF16_TO_FP32_Single 171
BF16_TO_FP32_Single 173
v_mac_f32 v[vgprValuC+60], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+61], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+62], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+63], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+64], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+65], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+66], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+67], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+68], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+69], v153, s[sgprBeta]
v_mac_f32 v[vgprValuC+70], v155, s[sgprBeta]
v_mac_f32 v[vgprValuC+71], v157, s[sgprBeta]
v_mac_f32 v[vgprValuC+72], v159, s[sgprBeta]
v_mac_f32 v[vgprValuC+73], v161, s[sgprBeta]
v_mac_f32 v[vgprValuC+74], v163, s[sgprBeta]
v_mac_f32 v[vgprValuC+75], v165, s[sgprBeta]
v_mac_f32 v[vgprValuC+76], v167, s[sgprBeta]
v_mac_f32 v[vgprValuC+77], v169, s[sgprBeta]
v_mac_f32 v[vgprValuC+78], v171, s[sgprBeta]
v_mac_f32 v[vgprValuC+79], v173, s[sgprBeta]

label_BiasAddrValid_End_2_7:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 60
FP32_TO_BF16_Single 61
FP32_TO_BF16_Single 62
FP32_TO_BF16_Single 63
FP32_TO_BF16_Single 64
FP32_TO_BF16_Single 65
FP32_TO_BF16_Single 66
FP32_TO_BF16_Single 67
FP32_TO_BF16_Single 68
FP32_TO_BF16_Single 69
FP32_TO_BF16_Single 70
FP32_TO_BF16_Single 71
FP32_TO_BF16_Single 72
FP32_TO_BF16_Single 73
FP32_TO_BF16_Single 74
FP32_TO_BF16_Single 75
FP32_TO_BF16_Single 76
FP32_TO_BF16_Single 77
FP32_TO_BF16_Single 78
FP32_TO_BF16_Single 79
buffer_store_short v60, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v61, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v62, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v63, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v64, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v65, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v66, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v67, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v68, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v69, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v70, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v71, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v72, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v73, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v74, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v75, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v76, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v77, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v78, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v79, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,5,4:vw1); (0,0,5,5:vw1); (0,0,5,6:vw1); (0,0,5,7:vw1); (0,1,5,0:vw1); (0,1,5,1:vw1); (0,1,5,2:vw1); (0,1,5,3:vw1); (0,1,5,4:vw1); (0,1,5,5:vw1); (0,1,5,6:vw1); (0,1,5,7:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck], v134     // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck], v136     // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck], v138     // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
_v_add_co_u32 v132, vcc, v128, 64                  // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck], v140     // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,4) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,5) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,6) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,7) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=0 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck], v150     // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,1) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=1
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck], v152     // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,2) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=2
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck], v154     // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,3) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=1 vc0=3
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck], v156     // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck+1], v156   // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,4) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck], v158     // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck+1], v158   // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,5) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=5
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck], v160     // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck+1], v160   // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,6) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=6
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck], v162     // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck+1], v162   // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,1,7) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=1 vc0=7
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck], v164     // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck+1], v164   // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset


/* (d1,vc1,d0,vc0)=(0,6,0,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck], v166     // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck+1], v166   // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck], v168     // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck+1], v168   // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck], v170     // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck+1], v170   // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck], v172     // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck+1], v172   // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 5, 4), (0, 0, 5, 5), (0, 0, 5, 6), (0, 0, 5, 7), (0, 1, 5, 0), (0, 1, 5, 1), (0, 1, 5, 2), (0, 1, 5, 3), (0, 1, 5, 4), (0, 1, 5, 5), (0, 1, 5, 6), (0, 1, 5, 7), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3)] */
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 153
BF16_TO_FP32_Single 155
BF16_TO_FP32_Single 157
BF16_TO_FP32_Single 159
BF16_TO_FP32_Single 161
BF16_TO_FP32_Single 163
BF16_TO_FP32_Single 165
BF16_TO_FP32_Single 167
BF16_TO_FP32_Single 169
BF16_TO_FP32_Single 171
BF16_TO_FP32_Single 173
v_mac_f32 v[vgprValuC+80], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+81], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+82], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+83], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+84], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+85], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+86], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+87], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+88], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+89], v153, s[sgprBeta]
v_mac_f32 v[vgprValuC+90], v155, s[sgprBeta]
v_mac_f32 v[vgprValuC+91], v157, s[sgprBeta]
v_mac_f32 v[vgprValuC+92], v159, s[sgprBeta]
v_mac_f32 v[vgprValuC+93], v161, s[sgprBeta]
v_mac_f32 v[vgprValuC+94], v163, s[sgprBeta]
v_mac_f32 v[vgprValuC+95], v165, s[sgprBeta]
v_mac_f32 v[vgprValuC+96], v167, s[sgprBeta]
v_mac_f32 v[vgprValuC+97], v169, s[sgprBeta]
v_mac_f32 v[vgprValuC+98], v171, s[sgprBeta]
v_mac_f32 v[vgprValuC+99], v173, s[sgprBeta]

label_BiasAddrValid_End_2_8:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 80
FP32_TO_BF16_Single 81
FP32_TO_BF16_Single 82
FP32_TO_BF16_Single 83
FP32_TO_BF16_Single 84
FP32_TO_BF16_Single 85
FP32_TO_BF16_Single 86
FP32_TO_BF16_Single 87
FP32_TO_BF16_Single 88
FP32_TO_BF16_Single 89
FP32_TO_BF16_Single 90
FP32_TO_BF16_Single 91
FP32_TO_BF16_Single 92
FP32_TO_BF16_Single 93
FP32_TO_BF16_Single 94
FP32_TO_BF16_Single 95
FP32_TO_BF16_Single 96
FP32_TO_BF16_Single 97
FP32_TO_BF16_Single 98
FP32_TO_BF16_Single 99
buffer_store_short v80, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v81, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v82, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v83, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v84, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v85, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v86, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v87, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v88, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v89, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v90, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v91, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v92, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v93, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v94, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v95, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v96, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v97, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v98, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v99, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #5 (d1,d0,vc1,vc0) = */
/*    (0,0,6,4:vw1); (0,0,6,5:vw1); (0,0,6,6:vw1); (0,0,6,7:vw1); (0,1,6,0:vw1); (0,1,6,1:vw1); (0,1,6,2:vw1); (0,1,6,3:vw1); (0,1,6,4:vw1); (0,1,6,5:vw1); (0,1,6,6:vw1); (0,1,6,7:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,7,4:vw1); (0,0,7,5:vw1); (0,0,7,6:vw1); (0,0,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,6,0,4) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck], v134     // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,5) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck], v136     // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,6) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck], v138     // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,7) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck], v140     // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,4) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[76:77], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[76:77], s[54:55], s[76:77]             // in0 && in1
_v_add_lshl_u32 v150, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck], v150     // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDC clip if OOB. offset
buffer_load_short_d16 v151, v150, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v150, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v150, vcc, s[sgprEdgeCheck+1], v150   // 
v_cndmask_b32 v150, -1, v150, s[76:77]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,5) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[78:79], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[78:79], s[54:55], s[78:79]             // in0 && in1
_v_add_lshl_u32 v152, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffC+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck], v152     // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDC clip if OOB. offset
buffer_load_short_d16 v153, v152, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v152, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v152, vcc, s[sgprStrideoffD+0], v152  // add 4* loop offset
v_add_co_u32 v152, vcc, s[sgprEdgeCheck+1], v152   // 
v_cndmask_b32 v152, -1, v152, s[78:79]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,6) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[80:81], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[80:81], s[54:55], s[80:81]             // in0 && in1
_v_add_lshl_u32 v154, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffC+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck], v154     // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDC clip if OOB. offset
buffer_load_short_d16 v155, v154, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v154, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v154, vcc, s[sgprStrideoffD+1], v154  // add 4* loop offset
v_add_co_u32 v154, vcc, s[sgprEdgeCheck+1], v154   // 
v_cndmask_b32 v154, -1, v154, s[80:81]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,1,7) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[82:83], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[82:83], s[54:55], s[82:83]             // in0 && in1
_v_add_lshl_u32 v156, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffC+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck], v156     // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDC clip if OOB. offset
buffer_load_short_d16 v157, v156, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v156, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v156, vcc, s[sgprStrideoffD+2], v156  // add 4* loop offset
v_add_co_u32 v156, vcc, s[sgprEdgeCheck+1], v156   // 
v_cndmask_b32 v156, -1, v156, s[82:83]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[84:85], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[84:85], s[54:55], s[84:85]             // in0 && in1
_v_add_lshl_u32 v158, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck], v158     // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDC clip if OOB. offset
buffer_load_short_d16 v159, v158, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v158, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v158, vcc, s[sgprEdgeCheck+1], v158   // 
v_cndmask_b32 v158, -1, v158, s[84:85]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[86:87], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[86:87], s[54:55], s[86:87]             // in0 && in1
_v_add_lshl_u32 v160, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffC+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck], v160     // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDC clip if OOB. offset
buffer_load_short_d16 v161, v160, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v160, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v160, vcc, s[sgprStrideoffD+0], v160  // add 4* loop offset
v_add_co_u32 v160, vcc, s[sgprEdgeCheck+1], v160   // 
v_cndmask_b32 v160, -1, v160, s[86:87]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[88:89], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[88:89], s[54:55], s[88:89]             // in0 && in1
_v_add_lshl_u32 v162, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffC+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck], v162     // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDC clip if OOB. offset
buffer_load_short_d16 v163, v162, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v162, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v162, vcc, s[sgprStrideoffD+1], v162  // add 4* loop offset
v_add_co_u32 v162, vcc, s[sgprEdgeCheck+1], v162   // 
v_cndmask_b32 v162, -1, v162, s[88:89]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[90:91], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[90:91], s[54:55], s[90:91]             // in0 && in1
_v_add_lshl_u32 v164, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffC+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck], v164     // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDC clip if OOB. offset
buffer_load_short_d16 v165, v164, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v164, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v164, vcc, s[sgprStrideoffD+2], v164  // add 4* loop offset
v_add_co_u32 v164, vcc, s[sgprEdgeCheck+1], v164   // 
v_cndmask_b32 v164, -1, v164, s[90:91]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,4) */
s_mov_b32 s54, 208                                  // coordOffset0 d0=0 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[92:93], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[92:93], s[54:55], s[92:93]             // in0 && in1
_v_add_lshl_u32 v166, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck], v166     // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDC clip if OOB. offset
buffer_load_short_d16 v167, v166, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v166, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v166, vcc, s[sgprEdgeCheck+1], v166   // 
v_cndmask_b32 v166, -1, v166, s[92:93]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,5) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[94:95], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[94:95], s[54:55], s[94:95]             // in0 && in1
_v_add_lshl_u32 v168, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffC+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck], v168     // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDC clip if OOB. offset
buffer_load_short_d16 v169, v168, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v168, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v168, vcc, s[sgprStrideoffD+0], v168  // add 4* loop offset
v_add_co_u32 v168, vcc, s[sgprEdgeCheck+1], v168   // 
v_cndmask_b32 v168, -1, v168, s[94:95]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,6) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[96:97], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[96:97], s[54:55], s[96:97]             // in0 && in1
_v_add_lshl_u32 v170, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffC+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck], v170     // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDC clip if OOB. offset
buffer_load_short_d16 v171, v170, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v170, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v170, vcc, s[sgprStrideoffD+1], v170  // add 4* loop offset
v_add_co_u32 v170, vcc, s[sgprEdgeCheck+1], v170   // 
v_cndmask_b32 v170, -1, v170, s[96:97]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,7) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[98:99], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[98:99], s[54:55], s[98:99]             // in0 && in1
_v_add_lshl_u32 v172, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffC+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck], v172     // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDC clip if OOB. offset
buffer_load_short_d16 v173, v172, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v172, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v172, vcc, s[sgprStrideoffD+2], v172  // add 4* loop offset
v_add_co_u32 v172, vcc, s[sgprEdgeCheck+1], v172   // 
v_cndmask_b32 v172, -1, v172, s[98:99]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 0, 6, 4), (0, 0, 6, 5), (0, 0, 6, 6), (0, 0, 6, 7), (0, 1, 6, 0), (0, 1, 6, 1), (0, 1, 6, 2), (0, 1, 6, 3), (0, 1, 6, 4), (0, 1, 6, 5), (0, 1, 6, 6), (0, 1, 6, 7), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 7, 4), (0, 0, 7, 5), (0, 0, 7, 6), (0, 0, 7, 7)] */
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 149
BF16_TO_FP32_Single 151
BF16_TO_FP32_Single 153
BF16_TO_FP32_Single 155
BF16_TO_FP32_Single 157
BF16_TO_FP32_Single 159
BF16_TO_FP32_Single 161
BF16_TO_FP32_Single 163
BF16_TO_FP32_Single 165
BF16_TO_FP32_Single 167
BF16_TO_FP32_Single 169
BF16_TO_FP32_Single 171
BF16_TO_FP32_Single 173
v_mac_f32 v[vgprValuC+100], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+101], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+102], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+103], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+104], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+105], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+106], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+107], v149, s[sgprBeta]
v_mac_f32 v[vgprValuC+108], v151, s[sgprBeta]
v_mac_f32 v[vgprValuC+109], v153, s[sgprBeta]
v_mac_f32 v[vgprValuC+110], v155, s[sgprBeta]
v_mac_f32 v[vgprValuC+111], v157, s[sgprBeta]
v_mac_f32 v[vgprValuC+112], v159, s[sgprBeta]
v_mac_f32 v[vgprValuC+113], v161, s[sgprBeta]
v_mac_f32 v[vgprValuC+114], v163, s[sgprBeta]
v_mac_f32 v[vgprValuC+115], v165, s[sgprBeta]
v_mac_f32 v[vgprValuC+116], v167, s[sgprBeta]
v_mac_f32 v[vgprValuC+117], v169, s[sgprBeta]
v_mac_f32 v[vgprValuC+118], v171, s[sgprBeta]
v_mac_f32 v[vgprValuC+119], v173, s[sgprBeta]

label_BiasAddrValid_End_2_9:

FP32_TO_BF16_7FFF

FP32_TO_BF16_Single 100
FP32_TO_BF16_Single 101
FP32_TO_BF16_Single 102
FP32_TO_BF16_Single 103
FP32_TO_BF16_Single 104
FP32_TO_BF16_Single 105
FP32_TO_BF16_Single 106
FP32_TO_BF16_Single 107
FP32_TO_BF16_Single 108
FP32_TO_BF16_Single 109
FP32_TO_BF16_Single 110
FP32_TO_BF16_Single 111
FP32_TO_BF16_Single 112
FP32_TO_BF16_Single 113
FP32_TO_BF16_Single 114
FP32_TO_BF16_Single 115
FP32_TO_BF16_Single 116
FP32_TO_BF16_Single 117
FP32_TO_BF16_Single 118
FP32_TO_BF16_Single 119
buffer_store_short v100, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v101, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v102, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v103, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v104, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v105, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v106, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v107, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v108, v150, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v109, v152, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v110, v154, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v111, v156, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v112, v158, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v113, v160, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v114, v162, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v115, v164, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v116, v166, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v117, v168, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v118, v170, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v119, v172, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #6 (d1,d0,vc1,vc0) = */
/*    (0,1,7,0:vw1); (0,1,7,1:vw1); (0,1,7,2:vw1); (0,1,7,3:vw1); (0,1,7,4:vw1); (0,1,7,5:vw1); (0,1,7,6:vw1); (0,1,7,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,7,1,0) */
s_mov_b32 s54, 224                                  // coordOffset0 d0=1 vc0=0
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[60:61], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v134, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck], v134     // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDC clip if OOB. offset
buffer_load_short_d16 v135, v134, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v134, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v134, vcc, s[sgprEdgeCheck+1], v134   // 
v_cndmask_b32 v134, -1, v134, s[60:61]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,1) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[62:63], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v136, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffC+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck], v136     // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDC clip if OOB. offset
buffer_load_short_d16 v137, v136, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v136, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v136, vcc, s[sgprStrideoffD+0], v136  // add 4* loop offset
v_add_co_u32 v136, vcc, s[sgprEdgeCheck+1], v136   // 
v_cndmask_b32 v136, -1, v136, s[62:63]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,2) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[64:65], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v138, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffC+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck], v138     // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDC clip if OOB. offset
buffer_load_short_d16 v139, v138, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v138, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v138, vcc, s[sgprStrideoffD+1], v138  // add 4* loop offset
v_add_co_u32 v138, vcc, s[sgprEdgeCheck+1], v138   // 
v_cndmask_b32 v138, -1, v138, s[64:65]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,3) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[66:67], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v140, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffC+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck], v140     // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v141, v140, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v140, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v140, vcc, s[sgprStrideoffD+2], v140  // add 4* loop offset
v_add_co_u32 v140, vcc, s[sgprEdgeCheck+1], v140   // 
v_cndmask_b32 v140, -1, v140, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,4) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=1 vc0=4
_v_add_co_u32 v132, vcc, v128, s54                 // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_co_u32 v133, vcc, v129, 16                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[68:69], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v142, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck], v142     // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDC clip if OOB. offset
buffer_load_short_d16 v143, v142, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v142, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v142, vcc, s[sgprEdgeCheck+1], v142   // 
v_cndmask_b32 v142, -1, v142, s[68:69]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,5) */
_v_add_co_u32 v133, vcc, v129, 20                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[70:71], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v144, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffC+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck], v144     // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDC clip if OOB. offset
buffer_load_short_d16 v145, v144, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v144, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v144, vcc, s[sgprStrideoffD+0], v144  // add 4* loop offset
v_add_co_u32 v144, vcc, s[sgprEdgeCheck+1], v144   // 
v_cndmask_b32 v144, -1, v144, s[70:71]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,6) */
_v_add_co_u32 v133, vcc, v129, 24                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[72:73], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v146, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffC+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck], v146     // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDC clip if OOB. offset
buffer_load_short_d16 v147, v146, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v146, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v146, vcc, s[sgprStrideoffD+1], v146  // add 4* loop offset
v_add_co_u32 v146, vcc, s[sgprEdgeCheck+1], v146   // 
v_cndmask_b32 v146, -1, v146, s[72:73]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,1,7) */
_v_add_co_u32 v133, vcc, v129, 28                  // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_cmp_lt_u32 s[54:55], v132, s[sgprSizeI]          // coord0 < size0
v_cmp_lt_u32 s[74:75], v133, s[sgprSizeJ]          // coord1 < size1
s_and_b64 s[74:75], s[54:55], s[74:75]             // in0 && in1
_v_add_lshl_u32 v148, v130, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffC+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck], v148     // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDC clip if OOB. offset
buffer_load_short_d16 v149, v148, s[sgprSrdC:sgprSrdC+3], 0, offen, offset:0 // load C for beta calc
_v_add_lshl_u32 v148, v131, v132, 0x1              // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_add_co_u32 v148, vcc, s[sgprStrideoffD+2], v148  // add 4* loop offset
v_add_co_u32 v148, vcc, s[sgprEdgeCheck+1], v148   // 
v_cndmask_b32 v148, -1, v148, s[74:75]             // LDD clip if OOB. offset

/* rC *= alpha batchEements=[(0, 1, 7, 0), (0, 1, 7, 1), (0, 1, 7, 2), (0, 1, 7, 3), (0, 1, 7, 4), (0, 1, 7, 5), (0, 1, 7, 6), (0, 1, 7, 7)] */
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C

/* apply mask, calc new C and issue writes */
BF16_TO_FP32_Single 135
BF16_TO_FP32_Single 137
BF16_TO_FP32_Single 139
BF16_TO_FP32_Single 141
BF16_TO_FP32_Single 143
BF16_TO_FP32_Single 145
BF16_TO_FP32_Single 147
BF16_TO_FP32_Single 149
v_mac_f32 v[vgprValuC+120], v135, s[sgprBeta]
v_mac_f32 v[vgprValuC+121], v137, s[sgprBeta]
v_mac_f32 v[vgprValuC+122], v139, s[sgprBeta]
v_mac_f32 v[vgprValuC+123], v141, s[sgprBeta]
v_mac_f32 v[vgprValuC+124], v143, s[sgprBeta]
v_mac_f32 v[vgprValuC+125], v145, s[sgprBeta]
v_mac_f32 v[vgprValuC+126], v147, s[sgprBeta]
v_mac_f32 v[vgprValuC+127], v149, s[sgprBeta]

label_BiasAddrValid_End_2_10:

FP32_TO_BF16_7FFF
FP32_TO_BF16_Single 120
FP32_TO_BF16_Single 121
FP32_TO_BF16_Single 122
FP32_TO_BF16_Single 123
FP32_TO_BF16_Single 124
FP32_TO_BF16_Single 125
FP32_TO_BF16_Single 126
FP32_TO_BF16_Single 127
buffer_store_short v120, v134, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v121, v136, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v122, v138, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v123, v140, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v124, v142, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v125, v144, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v126, v146, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
buffer_store_short v127, v148, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_branch label_GW_End_37                           // jump to end
label_GW_End_37:

label_0039:  /// KernelEnd
s_endpgm                                           // Kernel End
