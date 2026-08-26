crypto/des.initFeistelBox STEXT nosplit size=236 align=0x0 args=0x0 locals=0x0 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:85)	TEXT	crypto/des.initFeistelBox(SB), NOSPLIT|NOFRAME|ABIInternal, $0-0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:85)	FUNCDATA	$0, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:85)	FUNCDATA	$1, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:86)	XORL	AX, AX
	0x0002 00002 (/home/frank/arm-128base/src/crypto/des/block.go:86)	JMP	7
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/block.go:86)	INCQ	AX
	0x0007 00007 (/home/frank/arm-128base/src/crypto/des/block.go:86)	CMPQ	AX, $8
	0x000b 00011 (/home/frank/arm-128base/src/crypto/des/block.go:86)	JGE	235
	0x0011 00017 (/home/frank/arm-128base/src/crypto/des/block.go:87)	XORL	DX, DX
	0x0013 00019 (/home/frank/arm-128base/src/crypto/des/block.go:87)	JMP	24
	0x0015 00021 (/home/frank/arm-128base/src/crypto/des/block.go:87)	INCQ	DX
	0x0018 00024 (/home/frank/arm-128base/src/crypto/des/block.go:87)	CMPQ	DX, $4
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/block.go:87)	JGE	4
	0x001e 00030 (/home/frank/arm-128base/src/crypto/des/block.go:88)	XORL	BX, BX
	0x0020 00032 (/home/frank/arm-128base/src/crypto/des/block.go:88)	JMP	103
	0x0022 00034 (/home/frank/arm-128base/src/crypto/des/block.go:94)	MOVQ	DI, R9
	0x0025 00037 (/home/frank/arm-128base/src/crypto/des/block.go:94)	ANDL	$2, DI
	0x0028 00040 (/home/frank/arm-128base/src/crypto/des/block.go:94)	SHLQ	$4, DI
	0x002c 00044 (/home/frank/arm-128base/src/crypto/des/block.go:94)	MOVQ	R9, R10
	0x002f 00047 (/home/frank/arm-128base/src/crypto/des/block.go:94)	ANDL	$1, R9
	0x0033 00051 (/home/frank/arm-128base/src/crypto/des/block.go:94)	ORQ	R9, DI
	0x0036 00054 (/home/frank/arm-128base/src/crypto/des/block.go:95)	LEAQ	(BX)(BX*1), R9
	0x003a 00058 (/home/frank/arm-128base/src/crypto/des/block.go:96)	ORL	R9, DI
	0x003d 00061 (/home/frank/arm-128base/src/crypto/des/block.go:99)	LEAQ	(DX)(DX*1), R9
	0x0041 00065 (/home/frank/arm-128base/src/crypto/des/block.go:99)	SHRQ	$31, DX
	0x0045 00069 (/home/frank/arm-128base/src/crypto/des/block.go:99)	ORQ	R9, DX
	0x0048 00072 (/home/frank/arm-128base/src/crypto/des/block.go:101)	MOVQ	R8, AX
	0x004b 00075 (/home/frank/arm-128base/src/crypto/des/block.go:101)	SHLQ	$8, R8
	0x004f 00079 (/home/frank/arm-128base/src/crypto/des/block.go:101)	LEAQ	crypto/des.feistelBox(SB), R9
	0x0056 00086 (/home/frank/arm-128base/src/crypto/des/block.go:101)	ADDQ	R9, R8
	0x0059 00089 (/home/frank/arm-128base/src/crypto/des/block.go:101)	MOVBLZX	DIB, DI
	0x005d 00093 (/home/frank/arm-128base/src/crypto/des/block.go:101)	MOVL	DX, (R8)(DI*4)
	0x0061 00097 (/home/frank/arm-128base/src/crypto/des/block.go:88)	INCQ	BX
	0x0064 00100 (/home/frank/arm-128base/src/crypto/des/block.go:89)	MOVQ	R10, DX
	0x0067 00103 (/home/frank/arm-128base/src/crypto/des/block.go:88)	CMPQ	BX, $16
	0x006b 00107 (/home/frank/arm-128base/src/crypto/des/block.go:88)	JGE	21
	0x006d 00109 (/home/frank/arm-128base/src/crypto/des/block.go:89)	MOVQ	AX, CX
	0x0070 00112 (/home/frank/arm-128base/src/crypto/des/block.go:89)	SHLQ	$6, AX
	0x0074 00116 (/home/frank/arm-128base/src/crypto/des/block.go:89)	LEAQ	crypto/des.sBoxes(SB), SI
	0x007b 00123 (/home/frank/arm-128base/src/crypto/des/block.go:89)	ADDQ	SI, AX
	0x007e 00126 (/home/frank/arm-128base/src/crypto/des/block.go:89)	MOVQ	DX, DI
	0x0081 00129 (/home/frank/arm-128base/src/crypto/des/block.go:89)	SHLQ	$4, DX
	0x0085 00133 (/home/frank/arm-128base/src/crypto/des/block.go:89)	ADDQ	DX, AX
	0x0088 00136 (/home/frank/arm-128base/src/crypto/des/block.go:89)	MOVBLZX	(AX)(BX*1), AX
	0x008c 00140 (/home/frank/arm-128base/src/crypto/des/block.go:89)	LEAQ	-7(CX), DX
	0x0090 00144 (/home/frank/arm-128base/src/crypto/des/block.go:89)	NEGQ	DX
	0x0093 00147 (/home/frank/arm-128base/src/crypto/des/block.go:89)	SHLQ	$2, DX
	0x0097 00151 (/home/frank/arm-128base/src/crypto/des/block.go:86)	MOVQ	CX, R8
	0x009a 00154 (/home/frank/arm-128base/src/crypto/des/block.go:89)	MOVQ	DX, CX
	0x009d 00157 (/home/frank/arm-128base/src/crypto/des/block.go:89)	SHLQ	CX, AX
	0x00a0 00160 (/home/frank/arm-128base/src/crypto/des/block.go:90)	XCHGL	AX, AX
	0x00a1 00161 (/home/frank/arm-128base/src/crypto/des/block.go:78)	XORL	DX, DX
	0x00a3 00163 (/home/frank/arm-128base/src/crypto/des/block.go:78)	XORL	R9, R9
	0x00a6 00166 (/home/frank/arm-128base/src/crypto/des/block.go:78)	JMP	224
	0x00a8 00168 (/home/frank/arm-128base/src/crypto/des/block.go:78)	LEAQ	crypto/des.permutationFunction(SB), R10
	0x00af 00175 (/home/frank/arm-128base/src/crypto/des/block.go:78)	MOVBLZX	(R10)(R9*1), CX
	0x00b4 00180 (/home/frank/arm-128base/src/crypto/des/block.go:79)	MOVQ	AX, R11
	0x00b7 00183 (/home/frank/arm-128base/src/crypto/des/block.go:79)	SHRQ	CX, R11
	0x00ba 00186 (/home/frank/arm-128base/src/crypto/des/block.go:80)	LEAQ	-31(R9), R12
	0x00be 00190 (/home/frank/arm-128base/src/crypto/des/block.go:80)	NEGQ	R12
	0x00c1 00193 (/home/frank/arm-128base/src/crypto/des/block.go:79)	CMPB	CL, $64
	0x00c4 00196 (/home/frank/arm-128base/src/crypto/des/block.go:79)	SBBQ	R13, R13
	0x00c7 00199 (/home/frank/arm-128base/src/crypto/des/block.go:79)	ANDQ	R13, R11
	0x00ca 00202 (/home/frank/arm-128base/src/crypto/des/block.go:79)	ANDL	$1, R11
	0x00ce 00206 (/home/frank/arm-128base/src/crypto/des/block.go:80)	MOVQ	R12, CX
	0x00d1 00209 (/home/frank/arm-128base/src/crypto/des/block.go:80)	SHLQ	CX, R11
	0x00d4 00212 (/home/frank/arm-128base/src/crypto/des/block.go:80)	ORQ	R11, DX
	0x00d7 00215 (/home/frank/arm-128base/src/crypto/des/block.go:78)	INCQ	R9
	0x00da 00218 (/home/frank/arm-128base/src/crypto/des/block.go:78)	NOP
	0x00e0 00224 (/home/frank/arm-128base/src/crypto/des/block.go:78)	CMPQ	R9, $32
	0x00e4 00228 (/home/frank/arm-128base/src/crypto/des/block.go:78)	JLT	168
	0x00e6 00230 (/home/frank/arm-128base/src/crypto/des/block.go:78)	JMP	34
	0x00eb 00235 (/home/frank/arm-128base/src/crypto/des/block.go:105)	RET
	0x0000 31 c0 eb 03 48 ff c0 48 83 f8 08 0f 8d da 00 00  1...H..H........
	0x0010 00 31 d2 eb 03 48 ff c2 48 83 fa 04 7d e6 31 db  .1...H..H...}.1.
	0x0020 eb 45 49 89 f9 83 e7 02 48 c1 e7 04 4d 89 ca 41  .EI.....H...M..A
	0x0030 83 e1 01 4c 09 cf 4c 8d 0c 1b 44 09 cf 4c 8d 0c  ...L..L...D..L..
	0x0040 12 48 c1 ea 1f 4c 09 ca 4c 89 c0 49 c1 e0 08 4c  .H...L..L..I...L
	0x0050 8d 0d 00 00 00 00 4d 01 c8 40 0f b6 ff 41 89 14  ......M..@...A..
	0x0060 b8 48 ff c3 4c 89 d2 48 83 fb 10 7d a8 48 89 c1  .H..L..H...}.H..
	0x0070 48 c1 e0 06 48 8d 35 00 00 00 00 48 01 f0 48 89  H...H.5....H..H.
	0x0080 d7 48 c1 e2 04 48 01 d0 0f b6 04 18 48 8d 51 f9  .H...H......H.Q.
	0x0090 48 f7 da 48 c1 e2 02 49 89 c8 48 89 d1 48 d3 e0  H..H...I..H..H..
	0x00a0 90 31 d2 45 31 c9 eb 38 4c 8d 15 00 00 00 00 43  .1.E1..8L......C
	0x00b0 0f b6 0c 0a 49 89 c3 49 d3 eb 4d 8d 61 e1 49 f7  ....I..I..M.a.I.
	0x00c0 dc 80 f9 40 4d 19 ed 4d 21 eb 41 83 e3 01 4c 89  ...@M..M!.A...L.
	0x00d0 e1 49 d3 e3 4c 09 da 49 ff c1 66 0f 1f 44 00 00  .I..L..I..f..D..
	0x00e0 49 83 f9 20 7c c2 e9 37 ff ff ff c3              I.. |..7....
	rel 82+4 t=R_PCREL crypto/des.feistelBox+0
	rel 119+4 t=R_PCREL crypto/des.sBoxes+0
	rel 171+4 t=R_PCREL crypto/des.permutationFunction+0
crypto/des.permuteInitialBlock STEXT nosplit size=217 align=0x0 args=0x8 locals=0x0 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:109)	TEXT	crypto/des.permuteInitialBlock(SB), NOSPLIT|NOFRAME|ABIInternal, $0-8
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:109)	FUNCDATA	$0, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:109)	FUNCDATA	$1, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:109)	FUNCDATA	$5, crypto/des.permuteInitialBlock.arginfo1(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:109)	FUNCDATA	$6, crypto/des.permuteInitialBlock.argliveinfo(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:109)	PCDATA	$3, $1
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:111)	MOVQ	AX, CX
	0x0003 00003 (/home/frank/arm-128base/src/crypto/des/block.go:111)	SHRQ	$48, CX
	0x0007 00007 (/home/frank/arm-128base/src/crypto/des/block.go:112)	MOVQ	AX, DX
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/block.go:112)	SHLQ	$48, DX
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/block.go:113)	XORQ	CX, DX
	0x0011 00017 (/home/frank/arm-128base/src/crypto/des/block.go:113)	SHLQ	$48, CX
	0x0015 00021 (/home/frank/arm-128base/src/crypto/des/block.go:113)	XORQ	DX, CX
	0x0018 00024 (/home/frank/arm-128base/src/crypto/des/block.go:113)	MOVWLZX	AX, DX
	0x001b 00027 (/home/frank/arm-128base/src/crypto/des/block.go:113)	XORQ	CX, DX
	0x001e 00030 (/home/frank/arm-128base/src/crypto/des/block.go:113)	XORQ	DX, AX
	0x0021 00033 (/home/frank/arm-128base/src/crypto/des/block.go:117)	MOVL	$4278255360, CX
	0x0026 00038 (/home/frank/arm-128base/src/crypto/des/block.go:117)	ANDQ	AX, CX
	0x0029 00041 (/home/frank/arm-128base/src/crypto/des/block.go:118)	MOVQ	$71777214277877760, DX
	0x0033 00051 (/home/frank/arm-128base/src/crypto/des/block.go:118)	ANDQ	AX, DX
	0x0036 00054 (/home/frank/arm-128base/src/crypto/des/block.go:118)	XORQ	CX, DX
	0x0039 00057 (/home/frank/arm-128base/src/crypto/des/block.go:118)	MOVQ	AX, BX
	0x003c 00060 (/home/frank/arm-128base/src/crypto/des/block.go:118)	SHRQ	$24, BX
	0x0040 00064 (/home/frank/arm-128base/src/crypto/des/block.go:118)	MOVL	$4278255360, SI
	0x0045 00069 (/home/frank/arm-128base/src/crypto/des/block.go:118)	ANDQ	BX, SI
	0x0048 00072 (/home/frank/arm-128base/src/crypto/des/block.go:118)	XORQ	DX, SI
	0x004b 00075 (/home/frank/arm-128base/src/crypto/des/block.go:118)	SHLQ	$24, CX
	0x004f 00079 (/home/frank/arm-128base/src/crypto/des/block.go:118)	XORQ	SI, CX
	0x0052 00082 (/home/frank/arm-128base/src/crypto/des/block.go:118)	XORQ	CX, AX
	0x0055 00085 (/home/frank/arm-128base/src/crypto/des/block.go:131)	MOVQ	$1085086035472220160, CX
	0x005f 00095 (/home/frank/arm-128base/src/crypto/des/block.go:131)	ANDQ	AX, CX
	0x0062 00098 (/home/frank/arm-128base/src/crypto/des/block.go:132)	MOVQ	$264913582878960, DX
	0x006c 00108 (/home/frank/arm-128base/src/crypto/des/block.go:132)	ANDQ	AX, DX
	0x006f 00111 (/home/frank/arm-128base/src/crypto/des/block.go:133)	MOVQ	DX, BX
	0x0072 00114 (/home/frank/arm-128base/src/crypto/des/block.go:133)	XORQ	CX, DX
	0x0075 00117 (/home/frank/arm-128base/src/crypto/des/block.go:133)	SHRQ	$12, CX
	0x0079 00121 (/home/frank/arm-128base/src/crypto/des/block.go:133)	XORQ	DX, CX
	0x007c 00124 (/home/frank/arm-128base/src/crypto/des/block.go:133)	SHLQ	$12, BX
	0x0080 00128 (/home/frank/arm-128base/src/crypto/des/block.go:133)	XORQ	CX, BX
	0x0083 00131 (/home/frank/arm-128base/src/crypto/des/block.go:133)	XORQ	BX, AX
	0x0086 00134 (/home/frank/arm-128base/src/crypto/des/block.go:145)	MOVQ	$3674993371882992384, CX
	0x0090 00144 (/home/frank/arm-128base/src/crypto/des/block.go:145)	ANDQ	AX, CX
	0x0093 00147 (/home/frank/arm-128base/src/crypto/des/block.go:146)	MOVQ	$57421771435671756, DX
	0x009d 00157 (/home/frank/arm-128base/src/crypto/des/block.go:146)	ANDQ	AX, DX
	0x00a0 00160 (/home/frank/arm-128base/src/crypto/des/block.go:147)	MOVQ	DX, BX
	0x00a3 00163 (/home/frank/arm-128base/src/crypto/des/block.go:147)	XORQ	CX, DX
	0x00a6 00166 (/home/frank/arm-128base/src/crypto/des/block.go:147)	SHRQ	$6, CX
	0x00aa 00170 (/home/frank/arm-128base/src/crypto/des/block.go:147)	XORQ	DX, CX
	0x00ad 00173 (/home/frank/arm-128base/src/crypto/des/block.go:147)	SHLQ	$6, BX
	0x00b1 00177 (/home/frank/arm-128base/src/crypto/des/block.go:147)	XORQ	CX, BX
	0x00b4 00180 (/home/frank/arm-128base/src/crypto/des/block.go:147)	XORQ	BX, AX
	0x00b7 00183 (/home/frank/arm-128base/src/crypto/des/block.go:160)	MOVQ	$-6148914692668172971, CX
	0x00c1 00193 (/home/frank/arm-128base/src/crypto/des/block.go:160)	ANDQ	AX, CX
	0x00c4 00196 (/home/frank/arm-128base/src/crypto/des/block.go:161)	MOVQ	CX, DX
	0x00c7 00199 (/home/frank/arm-128base/src/crypto/des/block.go:161)	SHRQ	$33, CX
	0x00cb 00203 (/home/frank/arm-128base/src/crypto/des/block.go:161)	XORQ	DX, CX
	0x00ce 00206 (/home/frank/arm-128base/src/crypto/des/block.go:161)	SHLQ	$33, DX
	0x00d2 00210 (/home/frank/arm-128base/src/crypto/des/block.go:161)	XORQ	CX, DX
	0x00d5 00213 (/home/frank/arm-128base/src/crypto/des/block.go:161)	XORQ	DX, AX
	0x00d8 00216 (/home/frank/arm-128base/src/crypto/des/block.go:172)	RET
	0x0000 48 89 c1 48 c1 e9 30 48 89 c2 48 c1 e2 30 48 31  H..H..0H..H..0H1
	0x0010 ca 48 c1 e1 30 48 31 d1 0f b7 d0 48 31 ca 48 31  .H..0H1....H1.H1
	0x0020 d0 b9 00 ff 00 ff 48 21 c1 48 ba 00 00 00 00 ff  ......H!.H......
	0x0030 00 ff 00 48 21 c2 48 31 ca 48 89 c3 48 c1 eb 18  ...H!.H1.H..H...
	0x0040 be 00 ff 00 ff 48 21 de 48 31 d6 48 c1 e1 18 48  .....H!.H1.H...H
	0x0050 31 f1 48 31 c8 48 b9 00 00 0f 0f 00 00 0f 0f 48  1.H1.H.........H
	0x0060 21 c1 48 ba f0 f0 00 00 f0 f0 00 00 48 21 c2 48  !.H.........H!.H
	0x0070 89 d3 48 31 ca 48 c1 e9 0c 48 31 d1 48 c1 e3 0c  ..H1.H...H1.H...
	0x0080 48 31 cb 48 31 d8 48 b9 00 33 00 33 00 33 00 33  H1.H1.H..3.3.3.3
	0x0090 48 21 c1 48 ba cc 00 cc 00 cc 00 cc 00 48 21 c2  H!.H.........H!.
	0x00a0 48 89 d3 48 31 ca 48 c1 e9 06 48 31 d1 48 c1 e3  H..H1.H...H1.H..
	0x00b0 06 48 31 cb 48 31 d8 48 b9 55 55 55 55 aa aa aa  .H1.H1.H.UUUU...
	0x00c0 aa 48 21 c1 48 89 ca 48 c1 e9 21 48 31 d1 48 c1  .H!.H..H..!H1.H.
	0x00d0 e2 21 48 31 ca 48 31 d0 c3                       .!H1.H1..
crypto/des.permuteFinalBlock STEXT nosplit size=217 align=0x0 args=0x8 locals=0x0 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:177)	TEXT	crypto/des.permuteFinalBlock(SB), NOSPLIT|NOFRAME|ABIInternal, $0-8
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:177)	FUNCDATA	$0, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:177)	FUNCDATA	$1, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:177)	FUNCDATA	$5, crypto/des.permuteFinalBlock.arginfo1(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:177)	FUNCDATA	$6, crypto/des.permuteFinalBlock.argliveinfo(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:177)	PCDATA	$3, $1
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:180)	MOVQ	$-6148914692668172971, CX
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/block.go:180)	ANDQ	AX, CX
	0x000d 00013 (/home/frank/arm-128base/src/crypto/des/block.go:181)	MOVQ	CX, DX
	0x0010 00016 (/home/frank/arm-128base/src/crypto/des/block.go:181)	SHRQ	$33, CX
	0x0014 00020 (/home/frank/arm-128base/src/crypto/des/block.go:181)	XORQ	DX, CX
	0x0017 00023 (/home/frank/arm-128base/src/crypto/des/block.go:181)	SHLQ	$33, DX
	0x001b 00027 (/home/frank/arm-128base/src/crypto/des/block.go:181)	XORQ	CX, DX
	0x001e 00030 (/home/frank/arm-128base/src/crypto/des/block.go:181)	XORQ	DX, AX
	0x0021 00033 (/home/frank/arm-128base/src/crypto/des/block.go:183)	MOVQ	$3674993371882992384, CX
	0x002b 00043 (/home/frank/arm-128base/src/crypto/des/block.go:183)	ANDQ	AX, CX
	0x002e 00046 (/home/frank/arm-128base/src/crypto/des/block.go:184)	MOVQ	$57421771435671756, DX
	0x0038 00056 (/home/frank/arm-128base/src/crypto/des/block.go:184)	ANDQ	AX, DX
	0x003b 00059 (/home/frank/arm-128base/src/crypto/des/block.go:185)	MOVQ	DX, BX
	0x003e 00062 (/home/frank/arm-128base/src/crypto/des/block.go:185)	XORQ	CX, DX
	0x0041 00065 (/home/frank/arm-128base/src/crypto/des/block.go:185)	SHRQ	$6, CX
	0x0045 00069 (/home/frank/arm-128base/src/crypto/des/block.go:185)	XORQ	DX, CX
	0x0048 00072 (/home/frank/arm-128base/src/crypto/des/block.go:185)	SHLQ	$6, BX
	0x004c 00076 (/home/frank/arm-128base/src/crypto/des/block.go:185)	XORQ	CX, BX
	0x004f 00079 (/home/frank/arm-128base/src/crypto/des/block.go:185)	XORQ	BX, AX
	0x0052 00082 (/home/frank/arm-128base/src/crypto/des/block.go:187)	MOVQ	$1085086035472220160, CX
	0x005c 00092 (/home/frank/arm-128base/src/crypto/des/block.go:187)	ANDQ	AX, CX
	0x005f 00095 (/home/frank/arm-128base/src/crypto/des/block.go:188)	MOVQ	$264913582878960, DX
	0x0069 00105 (/home/frank/arm-128base/src/crypto/des/block.go:188)	ANDQ	AX, DX
	0x006c 00108 (/home/frank/arm-128base/src/crypto/des/block.go:189)	MOVQ	DX, BX
	0x006f 00111 (/home/frank/arm-128base/src/crypto/des/block.go:189)	XORQ	CX, DX
	0x0072 00114 (/home/frank/arm-128base/src/crypto/des/block.go:189)	SHRQ	$12, CX
	0x0076 00118 (/home/frank/arm-128base/src/crypto/des/block.go:189)	XORQ	DX, CX
	0x0079 00121 (/home/frank/arm-128base/src/crypto/des/block.go:189)	SHLQ	$12, BX
	0x007d 00125 (/home/frank/arm-128base/src/crypto/des/block.go:189)	XORQ	CX, BX
	0x0080 00128 (/home/frank/arm-128base/src/crypto/des/block.go:189)	XORQ	BX, AX
	0x0083 00131 (/home/frank/arm-128base/src/crypto/des/block.go:192)	MOVL	$4278255360, CX
	0x0088 00136 (/home/frank/arm-128base/src/crypto/des/block.go:192)	ANDQ	AX, CX
	0x008b 00139 (/home/frank/arm-128base/src/crypto/des/block.go:193)	MOVQ	$71777214277877760, DX
	0x0095 00149 (/home/frank/arm-128base/src/crypto/des/block.go:193)	ANDQ	AX, DX
	0x0098 00152 (/home/frank/arm-128base/src/crypto/des/block.go:193)	XORQ	CX, DX
	0x009b 00155 (/home/frank/arm-128base/src/crypto/des/block.go:193)	MOVQ	AX, BX
	0x009e 00158 (/home/frank/arm-128base/src/crypto/des/block.go:193)	SHRQ	$24, BX
	0x00a2 00162 (/home/frank/arm-128base/src/crypto/des/block.go:193)	MOVL	$4278255360, SI
	0x00a7 00167 (/home/frank/arm-128base/src/crypto/des/block.go:193)	ANDQ	BX, SI
	0x00aa 00170 (/home/frank/arm-128base/src/crypto/des/block.go:193)	XORQ	DX, SI
	0x00ad 00173 (/home/frank/arm-128base/src/crypto/des/block.go:193)	SHLQ	$24, CX
	0x00b1 00177 (/home/frank/arm-128base/src/crypto/des/block.go:193)	XORQ	SI, CX
	0x00b4 00180 (/home/frank/arm-128base/src/crypto/des/block.go:193)	XORQ	CX, AX
	0x00b7 00183 (/home/frank/arm-128base/src/crypto/des/block.go:195)	MOVQ	AX, CX
	0x00ba 00186 (/home/frank/arm-128base/src/crypto/des/block.go:195)	SHRQ	$48, CX
	0x00be 00190 (/home/frank/arm-128base/src/crypto/des/block.go:196)	MOVQ	AX, DX
	0x00c1 00193 (/home/frank/arm-128base/src/crypto/des/block.go:196)	SHLQ	$48, DX
	0x00c5 00197 (/home/frank/arm-128base/src/crypto/des/block.go:197)	XORQ	CX, DX
	0x00c8 00200 (/home/frank/arm-128base/src/crypto/des/block.go:197)	SHLQ	$48, CX
	0x00cc 00204 (/home/frank/arm-128base/src/crypto/des/block.go:197)	XORQ	DX, CX
	0x00cf 00207 (/home/frank/arm-128base/src/crypto/des/block.go:197)	MOVWLZX	AX, DX
	0x00d2 00210 (/home/frank/arm-128base/src/crypto/des/block.go:197)	XORQ	CX, DX
	0x00d5 00213 (/home/frank/arm-128base/src/crypto/des/block.go:197)	XORQ	DX, AX
	0x00d8 00216 (/home/frank/arm-128base/src/crypto/des/block.go:198)	RET
	0x0000 48 b9 55 55 55 55 aa aa aa aa 48 21 c1 48 89 ca  H.UUUU....H!.H..
	0x0010 48 c1 e9 21 48 31 d1 48 c1 e2 21 48 31 ca 48 31  H..!H1.H..!H1.H1
	0x0020 d0 48 b9 00 33 00 33 00 33 00 33 48 21 c1 48 ba  .H..3.3.3.3H!.H.
	0x0030 cc 00 cc 00 cc 00 cc 00 48 21 c2 48 89 d3 48 31  ........H!.H..H1
	0x0040 ca 48 c1 e9 06 48 31 d1 48 c1 e3 06 48 31 cb 48  .H...H1.H...H1.H
	0x0050 31 d8 48 b9 00 00 0f 0f 00 00 0f 0f 48 21 c1 48  1.H.........H!.H
	0x0060 ba f0 f0 00 00 f0 f0 00 00 48 21 c2 48 89 d3 48  .........H!.H..H
	0x0070 31 ca 48 c1 e9 0c 48 31 d1 48 c1 e3 0c 48 31 cb  1.H...H1.H...H1.
	0x0080 48 31 d8 b9 00 ff 00 ff 48 21 c1 48 ba 00 00 00  H1......H!.H....
	0x0090 00 ff 00 ff 00 48 21 c2 48 31 ca 48 89 c3 48 c1  .....H!.H1.H..H.
	0x00a0 eb 18 be 00 ff 00 ff 48 21 de 48 31 d6 48 c1 e1  .......H!.H1.H..
	0x00b0 18 48 31 f1 48 31 c8 48 89 c1 48 c1 e9 30 48 89  .H1.H1.H..H..0H.
	0x00c0 c2 48 c1 e2 30 48 31 ca 48 c1 e1 30 48 31 d1 0f  .H..0H1.H..0H1..
	0x00d0 b7 d0 48 31 ca 48 31 d0 c3                       ..H1.H1..
crypto/des.ksRotate STEXT size=165 align=0x0 args=0x8 locals=0x20 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:203)	TEXT	crypto/des.ksRotate(SB), ABIInternal, $32-8
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:203)	CMPQ	SP, 16(R14)
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/block.go:203)	PCDATA	$0, $-2
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/block.go:203)	JLS	143
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/block.go:203)	PCDATA	$0, $-1
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/block.go:203)	PUSHQ	BP
	0x000b 00011 (/home/frank/arm-128base/src/crypto/des/block.go:203)	MOVQ	SP, BP
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/block.go:203)	SUBQ	$24, SP
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/block.go:203)	FUNCDATA	$0, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/block.go:203)	FUNCDATA	$1, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/block.go:203)	FUNCDATA	$5, crypto/des.ksRotate.arginfo1(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/block.go:203)	FUNCDATA	$6, crypto/des.ksRotate.argliveinfo(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/block.go:203)	PCDATA	$3, $1
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/block.go:206)	MOVL	AX, crypto/des.in+40(SP)
	0x0016 00022 (/home/frank/arm-128base/src/crypto/des/block.go:206)	PCDATA	$3, $-1
	0x0016 00022 (/home/frank/arm-128base/src/crypto/des/block.go:204)	LEAQ	type:uint32(SB), AX
	0x001d 00029 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVL	$16, BX
	0x0022 00034 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVL	BX, CX
	0x0024 00036 (/home/frank/arm-128base/src/crypto/des/block.go:204)	PCDATA	$1, $0
	0x0024 00036 (/home/frank/arm-128base/src/crypto/des/block.go:204)	CALL	runtime.makeslice(SB)
	0x0029 00041 (/home/frank/arm-128base/src/crypto/des/block.go:206)	MOVL	crypto/des.in+40(SP), DX
	0x002d 00045 (/home/frank/arm-128base/src/crypto/des/block.go:206)	XORL	SI, SI
	0x002f 00047 (/home/frank/arm-128base/src/crypto/des/block.go:206)	JMP	128
	0x0031 00049 (/home/frank/arm-128base/src/crypto/des/block.go:208)	LEAQ	crypto/des.ksRotations(SB), DI
	0x0038 00056 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVBLZX	(DI)(SI*1), CX
	0x003c 00060 (/home/frank/arm-128base/src/crypto/des/block.go:208)	LEAL	4(CX), R8
	0x0040 00064 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	CX, BX
	0x0042 00066 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	R8, CX
	0x0045 00069 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	DX, R9
	0x0048 00072 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SHLL	CX, R9
	0x004b 00075 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SHLL	$4, DX
	0x004e 00078 (/home/frank/arm-128base/src/crypto/des/block.go:209)	MOVL	BX, R10
	0x0051 00081 (/home/frank/arm-128base/src/crypto/des/block.go:209)	NEGL	BX
	0x0053 00083 (/home/frank/arm-128base/src/crypto/des/block.go:209)	MOVL	BX, CX
	0x0055 00085 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SHRL	CX, DX
	0x0057 00087 (/home/frank/arm-128base/src/crypto/des/block.go:209)	ADDL	$-32, R10
	0x005b 00091 (/home/frank/arm-128base/src/crypto/des/block.go:209)	NEGL	R10
	0x005e 00094 (/home/frank/arm-128base/src/crypto/des/block.go:208)	CMPB	R8B, $32
	0x0062 00098 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SBBL	R8, R8
	0x0065 00101 (/home/frank/arm-128base/src/crypto/des/block.go:208)	ANDL	R8, R9
	0x0068 00104 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SHRL	$4, R9
	0x006c 00108 (/home/frank/arm-128base/src/crypto/des/block.go:209)	CMPB	R10B, $32
	0x0070 00112 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SBBL	R8, R8
	0x0073 00115 (/home/frank/arm-128base/src/crypto/des/block.go:209)	ANDL	R8, DX
	0x0076 00118 (/home/frank/arm-128base/src/crypto/des/block.go:210)	ORL	R9, DX
	0x0079 00121 (/home/frank/arm-128base/src/crypto/des/block.go:210)	MOVL	DX, (AX)(SI*4)
	0x007c 00124 (/home/frank/arm-128base/src/crypto/des/block.go:206)	INCQ	SI
	0x007f 00127 (/home/frank/arm-128base/src/crypto/des/block.go:206)	NOP
	0x0080 00128 (/home/frank/arm-128base/src/crypto/des/block.go:206)	CMPQ	SI, $16
	0x0084 00132 (/home/frank/arm-128base/src/crypto/des/block.go:206)	JLT	49
	0x0086 00134 (/home/frank/arm-128base/src/crypto/des/block.go:213)	MOVL	$16, BX
	0x008b 00139 (/home/frank/arm-128base/src/crypto/des/block.go:213)	MOVL	BX, CX
	0x008d 00141 (/home/frank/arm-128base/src/crypto/des/block.go:213)	LEAVEQ
	0x008e 00142 (/home/frank/arm-128base/src/crypto/des/block.go:213)	RET
	0x008f 00143 (/home/frank/arm-128base/src/crypto/des/block.go:213)	NOP
	0x008f 00143 (/home/frank/arm-128base/src/crypto/des/block.go:203)	PCDATA	$1, $-1
	0x008f 00143 (/home/frank/arm-128base/src/crypto/des/block.go:203)	PCDATA	$0, $-2
	0x008f 00143 (/home/frank/arm-128base/src/crypto/des/block.go:203)	MOVL	AX, 8(SP)
	0x0093 00147 (/home/frank/arm-128base/src/crypto/des/block.go:203)	CALL	runtime.morestack_noctxt(SB)
	0x0098 00152 (/home/frank/arm-128base/src/crypto/des/block.go:203)	PCDATA	$0, $-1
	0x0098 00152 (/home/frank/arm-128base/src/crypto/des/block.go:203)	MOVL	8(SP), AX
	0x009c 00156 (/home/frank/arm-128base/src/crypto/des/block.go:203)	NOP
	0x00a0 00160 (/home/frank/arm-128base/src/crypto/des/block.go:203)	JMP	0
	0x0000 49 3b 66 10 0f 86 85 00 00 00 55 48 89 e5 48 83  I;f.......UH..H.
	0x0010 ec 18 89 44 24 28 48 8d 05 00 00 00 00 bb 10 00  ...D$(H.........
	0x0020 00 00 89 d9 e8 00 00 00 00 8b 54 24 28 31 f6 eb  ..........T$(1..
	0x0030 4f 48 8d 3d 00 00 00 00 0f b6 0c 37 44 8d 41 04  OH.=.......7D.A.
	0x0040 89 cb 44 89 c1 41 89 d1 41 d3 e1 c1 e2 04 41 89  ..D..A..A.....A.
	0x0050 da f7 db 89 d9 d3 ea 41 83 c2 e0 41 f7 da 41 80  .......A...A..A.
	0x0060 f8 20 45 19 c0 45 21 c1 41 c1 e9 04 41 80 fa 20  . E..E!.A...A.. 
	0x0070 45 19 c0 44 21 c2 44 09 ca 89 14 b0 48 ff c6 90  E..D!.D.....H...
	0x0080 48 83 fe 10 7c ab bb 10 00 00 00 89 d9 c9 c3 89  H...|...........
	0x0090 44 24 08 e8 00 00 00 00 8b 44 24 08 0f 1f 40 00  D$.......D$...@.
	0x00a0 e9 5b ff ff ff                                   .[...
	rel 25+4 t=R_PCREL type:uint32+0
	rel 37+4 t=R_CALL runtime.makeslice+0
	rel 52+4 t=R_PCREL crypto/des.ksRotations+0
	rel 148+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.(*desCipher).generateSubkeys STEXT size=735 align=0x0 args=0x20 locals=0x98 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:217)	TEXT	crypto/des.(*desCipher).generateSubkeys(SB), ABIInternal, $152-32
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:217)	LEAQ	-24(SP), R12
	0x0005 00005 (/home/frank/arm-128base/src/crypto/des/block.go:217)	CMPQ	R12, 16(R14)
	0x0009 00009 (/home/frank/arm-128base/src/crypto/des/block.go:217)	PCDATA	$0, $-2
	0x0009 00009 (/home/frank/arm-128base/src/crypto/des/block.go:217)	JLS	685
	0x000f 00015 (/home/frank/arm-128base/src/crypto/des/block.go:217)	PCDATA	$0, $-1
	0x000f 00015 (/home/frank/arm-128base/src/crypto/des/block.go:217)	PUSHQ	BP
	0x0010 00016 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	SP, BP
	0x0013 00019 (/home/frank/arm-128base/src/crypto/des/block.go:217)	SUBQ	$144, SP
	0x001a 00026 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	BX, crypto/des.keyBytes+168(FP)
	0x0022 00034 (/home/frank/arm-128base/src/crypto/des/block.go:217)	FUNCDATA	$0, gclocals·TswRR9Pia9Wsluv5u1sUnA==(SB)
	0x0022 00034 (/home/frank/arm-128base/src/crypto/des/block.go:217)	FUNCDATA	$1, gclocals·J26BEvPExEQhJvjp9E8Whg==(SB)
	0x0022 00034 (/home/frank/arm-128base/src/crypto/des/block.go:217)	FUNCDATA	$5, crypto/des.(*desCipher).generateSubkeys.arginfo1(SB)
	0x0022 00034 (/home/frank/arm-128base/src/crypto/des/block.go:217)	FUNCDATA	$6, crypto/des.(*desCipher).generateSubkeys.argliveinfo(SB)
	0x0022 00034 (/home/frank/arm-128base/src/crypto/des/block.go:217)	PCDATA	$3, $1
	0x0022 00034 (/home/frank/arm-128base/src/crypto/des/block.go:218)	XCHGL	AX, AX
	0x0023 00035 (<unknown line number>)	NOP
	0x0023 00035 (/home/frank/arm-128base/src/sync/atomic/type.go:19)	MOVL	crypto/des.feistelBoxOnce(SB), DX
	0x0029 00041 (/home/frank/arm-128base/src/sync/atomic/type.go:19)	TESTL	DX, DX
	0x002b 00043 (/home/frank/arm-128base/src/sync/once.go:67)	JNE	112
	0x002d 00045 (/home/frank/arm-128base/src/sync/once.go:67)	MOVQ	AX, crypto/des.c+160(SP)
	0x0035 00053 (/home/frank/arm-128base/src/sync/once.go:67)	MOVQ	CX, crypto/des.keyBytes+176(SP)
	0x003d 00061 (/home/frank/arm-128base/src/sync/once.go:67)	MOVQ	BX, crypto/des.keyBytes+168(SP)
	0x0045 00069 (/home/frank/arm-128base/src/sync/once.go:67)	PCDATA	$3, $2
	0x0045 00069 (/home/frank/arm-128base/src/sync/once.go:69)	LEAQ	crypto/des.feistelBoxOnce(SB), AX
	0x004c 00076 (/home/frank/arm-128base/src/sync/once.go:69)	LEAQ	crypto/des.initFeistelBox·f(SB), BX
	0x0053 00083 (/home/frank/arm-128base/src/sync/once.go:69)	PCDATA	$1, $0
	0x0053 00083 (/home/frank/arm-128base/src/sync/once.go:69)	CALL	sync.(*Once).doSlow(SB)
	0x0058 00088 (/home/frank/arm-128base/src/crypto/des/block.go:233)	MOVQ	crypto/des.c+160(SP), AX
	0x0060 00096 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:121)	MOVQ	crypto/des.keyBytes+176(SP), CX
	0x0068 00104 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	MOVQ	crypto/des.keyBytes+168(SP), BX
	0x0070 00112 (/home/frank/arm-128base/src/sync/once.go:69)	PCDATA	$3, $1
	0x0070 00112 (/home/frank/arm-128base/src/crypto/des/block.go:221)	XCHGL	AX, AX
	0x0071 00113 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:121)	CMPQ	CX, $7
	0x0075 00117 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:121)	JLS	679
	0x007b 00123 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	MOVQ	(BX), DX
	0x007e 00126 (/home/frank/arm-128base/src/crypto/des/block.go:222)	XCHGL	AX, AX
	0x007f 00127 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	BSWAPQ	DX
	0x0082 00130 (/home/frank/arm-128base/src/crypto/des/block.go:78)	XORL	BX, BX
	0x0084 00132 (/home/frank/arm-128base/src/crypto/des/block.go:78)	XORL	SI, SI
	0x0086 00134 (/home/frank/arm-128base/src/crypto/des/block.go:78)	JMP	185
	0x0088 00136 (/home/frank/arm-128base/src/crypto/des/block.go:78)	LEAQ	crypto/des.permutedChoice1(SB), DI
	0x008f 00143 (/home/frank/arm-128base/src/crypto/des/block.go:78)	MOVBLZX	(BX)(DI*1), CX
	0x0093 00147 (/home/frank/arm-128base/src/crypto/des/block.go:79)	MOVQ	DX, R8
	0x0096 00150 (/home/frank/arm-128base/src/crypto/des/block.go:79)	SHRQ	CX, R8
	0x0099 00153 (/home/frank/arm-128base/src/crypto/des/block.go:80)	LEAQ	-55(BX), R9
	0x009d 00157 (/home/frank/arm-128base/src/crypto/des/block.go:80)	NEGQ	R9
	0x00a0 00160 (/home/frank/arm-128base/src/crypto/des/block.go:79)	CMPB	CL, $64
	0x00a3 00163 (/home/frank/arm-128base/src/crypto/des/block.go:79)	SBBQ	R10, R10
	0x00a6 00166 (/home/frank/arm-128base/src/crypto/des/block.go:79)	ANDQ	R10, R8
	0x00a9 00169 (/home/frank/arm-128base/src/crypto/des/block.go:79)	ANDL	$1, R8
	0x00ad 00173 (/home/frank/arm-128base/src/crypto/des/block.go:80)	MOVQ	R9, CX
	0x00b0 00176 (/home/frank/arm-128base/src/crypto/des/block.go:80)	SHLQ	CX, R8
	0x00b3 00179 (/home/frank/arm-128base/src/crypto/des/block.go:80)	ORQ	R8, SI
	0x00b6 00182 (/home/frank/arm-128base/src/crypto/des/block.go:78)	INCQ	BX
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/block.go:78)	CMPQ	BX, $56
	0x00bd 00189 (/home/frank/arm-128base/src/crypto/des/block.go:78)	JLT	136
	0x00bf 00191 (<unknown line number>)	NOP
	0x00bf 00191 (/home/frank/arm-128base/src/crypto/des/block.go:204)	LEAQ	crypto/des..autotmp_46+80(SP), DX
	0x00c4 00196 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVUPS	X15, (DX)
	0x00c8 00200 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVUPS	X15, 16(DX)
	0x00cd 00205 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVUPS	X15, 32(DX)
	0x00d2 00210 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVUPS	X15, 48(DX)
	0x00d7 00215 (/home/frank/arm-128base/src/crypto/des/block.go:225)	MOVQ	SI, DX
	0x00da 00218 (/home/frank/arm-128base/src/crypto/des/block.go:225)	SHRQ	$28, SI
	0x00de 00222 (/home/frank/arm-128base/src/crypto/des/block.go:206)	XORL	BX, BX
	0x00e0 00224 (/home/frank/arm-128base/src/crypto/des/block.go:206)	JMP	308
	0x00e2 00226 (/home/frank/arm-128base/src/crypto/des/block.go:208)	LEAQ	crypto/des.ksRotations(SB), DI
	0x00e9 00233 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVBLZX	(DI)(BX*1), CX
	0x00ed 00237 (/home/frank/arm-128base/src/crypto/des/block.go:208)	LEAL	4(CX), R8
	0x00f1 00241 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	CX, R9
	0x00f4 00244 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	R8, CX
	0x00f7 00247 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	SI, R10
	0x00fa 00250 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SHLL	CX, R10
	0x00fd 00253 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SHLL	$4, SI
	0x0100 00256 (/home/frank/arm-128base/src/crypto/des/block.go:209)	MOVL	R9, R11
	0x0103 00259 (/home/frank/arm-128base/src/crypto/des/block.go:209)	NEGL	R9
	0x0106 00262 (/home/frank/arm-128base/src/crypto/des/block.go:209)	MOVL	R9, CX
	0x0109 00265 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SHRL	CX, SI
	0x010b 00267 (/home/frank/arm-128base/src/crypto/des/block.go:209)	LEAL	-32(R11), R9
	0x010f 00271 (/home/frank/arm-128base/src/crypto/des/block.go:209)	NEGL	R9
	0x0112 00274 (/home/frank/arm-128base/src/crypto/des/block.go:208)	CMPB	R8B, $32
	0x0116 00278 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SBBL	R8, R8
	0x0119 00281 (/home/frank/arm-128base/src/crypto/des/block.go:208)	ANDL	R8, R10
	0x011c 00284 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SHRL	$4, R10
	0x0120 00288 (/home/frank/arm-128base/src/crypto/des/block.go:209)	CMPB	R9B, $32
	0x0124 00292 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SBBL	R8, R8
	0x0127 00295 (/home/frank/arm-128base/src/crypto/des/block.go:209)	ANDL	R8, SI
	0x012a 00298 (/home/frank/arm-128base/src/crypto/des/block.go:210)	ORL	R10, SI
	0x012d 00301 (/home/frank/arm-128base/src/crypto/des/block.go:210)	MOVL	SI, crypto/des..autotmp_46+80(SP)(BX*4)
	0x0131 00305 (/home/frank/arm-128base/src/crypto/des/block.go:206)	INCQ	BX
	0x0134 00308 (/home/frank/arm-128base/src/crypto/des/block.go:206)	CMPQ	BX, $16
	0x0138 00312 (/home/frank/arm-128base/src/crypto/des/block.go:206)	JLT	226
	0x013a 00314 (/home/frank/arm-128base/src/crypto/des/block.go:226)	SHLQ	$4, DX
	0x013e 00318 (<unknown line number>)	NOP
	0x013e 00318 (/home/frank/arm-128base/src/crypto/des/block.go:204)	LEAQ	crypto/des..autotmp_47+16(SP), BX
	0x0143 00323 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVUPS	X15, (BX)
	0x0147 00327 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVUPS	X15, 16(BX)
	0x014c 00332 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVUPS	X15, 32(BX)
	0x0151 00337 (/home/frank/arm-128base/src/crypto/des/block.go:204)	MOVUPS	X15, 48(BX)
	0x0156 00342 (/home/frank/arm-128base/src/crypto/des/block.go:226)	SHRL	$4, DX
	0x0159 00345 (/home/frank/arm-128base/src/crypto/des/block.go:206)	XORL	BX, BX
	0x015b 00347 (/home/frank/arm-128base/src/crypto/des/block.go:206)	JMP	426
	0x015d 00349 (/home/frank/arm-128base/src/crypto/des/block.go:208)	LEAQ	crypto/des.ksRotations(SB), SI
	0x0164 00356 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVBLZX	(BX)(SI*1), CX
	0x0168 00360 (/home/frank/arm-128base/src/crypto/des/block.go:208)	LEAL	4(CX), DI
	0x016b 00363 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	CX, R8
	0x016e 00366 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	DI, CX
	0x0170 00368 (/home/frank/arm-128base/src/crypto/des/block.go:208)	MOVL	DX, R9
	0x0173 00371 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SHLL	CX, R9
	0x0176 00374 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SHLL	$4, DX
	0x0179 00377 (/home/frank/arm-128base/src/crypto/des/block.go:209)	MOVL	R8, R10
	0x017c 00380 (/home/frank/arm-128base/src/crypto/des/block.go:209)	NEGL	R8
	0x017f 00383 (/home/frank/arm-128base/src/crypto/des/block.go:209)	MOVL	R8, CX
	0x0182 00386 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SHRL	CX, DX
	0x0184 00388 (/home/frank/arm-128base/src/crypto/des/block.go:209)	LEAL	-32(R10), R8
	0x0188 00392 (/home/frank/arm-128base/src/crypto/des/block.go:209)	NEGL	R8
	0x018b 00395 (/home/frank/arm-128base/src/crypto/des/block.go:208)	CMPB	DIB, $32
	0x018f 00399 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SBBL	DI, DI
	0x0191 00401 (/home/frank/arm-128base/src/crypto/des/block.go:208)	ANDL	DI, R9
	0x0194 00404 (/home/frank/arm-128base/src/crypto/des/block.go:208)	SHRL	$4, R9
	0x0198 00408 (/home/frank/arm-128base/src/crypto/des/block.go:209)	CMPB	R8B, $32
	0x019c 00412 (/home/frank/arm-128base/src/crypto/des/block.go:209)	SBBL	DI, DI
	0x019e 00414 (/home/frank/arm-128base/src/crypto/des/block.go:209)	ANDL	DI, DX
	0x01a0 00416 (/home/frank/arm-128base/src/crypto/des/block.go:210)	ORL	R9, DX
	0x01a3 00419 (/home/frank/arm-128base/src/crypto/des/block.go:210)	MOVL	DX, crypto/des..autotmp_47+16(SP)(BX*4)
	0x01a7 00423 (/home/frank/arm-128base/src/crypto/des/block.go:206)	INCQ	BX
	0x01aa 00426 (/home/frank/arm-128base/src/crypto/des/block.go:206)	CMPQ	BX, $16
	0x01ae 00430 (/home/frank/arm-128base/src/crypto/des/block.go:206)	JLT	349
	0x01b0 00432 (/home/frank/arm-128base/src/crypto/des/block.go:229)	XORL	DX, DX
	0x01b2 00434 (/home/frank/arm-128base/src/crypto/des/block.go:229)	JMP	586
	0x01b7 00439 (/home/frank/arm-128base/src/crypto/des/block.go:233)	TESTB	AL, (AX)
	0x01b9 00441 (/home/frank/arm-128base/src/crypto/des/block.go:241)	MOVQ	SI, BX
	0x01bc 00444 (/home/frank/arm-128base/src/crypto/des/block.go:241)	SHRQ	$6, SI
	0x01c0 00448 (/home/frank/arm-128base/src/crypto/des/block.go:241)	MOVBLZX	SIB, SI
	0x01c4 00452 (/home/frank/arm-128base/src/crypto/des/block.go:242)	MOVQ	BX, DI
	0x01c7 00455 (/home/frank/arm-128base/src/crypto/des/block.go:242)	SHRQ	$10, BX
	0x01cb 00459 (/home/frank/arm-128base/src/crypto/des/block.go:242)	ANDL	$65280, BX
	0x01d1 00465 (/home/frank/arm-128base/src/crypto/des/block.go:241)	ORQ	SI, BX
	0x01d4 00468 (/home/frank/arm-128base/src/crypto/des/block.go:243)	MOVQ	DI, SI
	0x01d7 00471 (/home/frank/arm-128base/src/crypto/des/block.go:243)	SHRQ	$14, DI
	0x01db 00475 (/home/frank/arm-128base/src/crypto/des/block.go:243)	ANDL	$16711680, DI
	0x01e1 00481 (/home/frank/arm-128base/src/crypto/des/block.go:242)	ORQ	BX, DI
	0x01e4 00484 (/home/frank/arm-128base/src/crypto/des/block.go:244)	MOVQ	SI, BX
	0x01e7 00487 (/home/frank/arm-128base/src/crypto/des/block.go:244)	SHRQ	$18, SI
	0x01eb 00491 (/home/frank/arm-128base/src/crypto/des/block.go:244)	MOVL	$4278190080, R8
	0x01f1 00497 (/home/frank/arm-128base/src/crypto/des/block.go:244)	ANDQ	R8, SI
	0x01f4 00500 (/home/frank/arm-128base/src/crypto/des/block.go:243)	ORQ	DI, SI
	0x01f7 00503 (/home/frank/arm-128base/src/crypto/des/block.go:245)	MOVBLZX	BL, DI
	0x01fa 00506 (/home/frank/arm-128base/src/crypto/des/block.go:245)	SHLQ	$32, DI
	0x01fe 00510 (/home/frank/arm-128base/src/crypto/des/block.go:244)	ORQ	SI, DI
	0x0201 00513 (/home/frank/arm-128base/src/crypto/des/block.go:246)	MOVQ	BX, SI
	0x0204 00516 (/home/frank/arm-128base/src/crypto/des/block.go:246)	SHLQ	$28, BX
	0x0208 00520 (/home/frank/arm-128base/src/crypto/des/block.go:246)	MOVQ	$280375465082880, R9
	0x0212 00530 (/home/frank/arm-128base/src/crypto/des/block.go:246)	ANDQ	R9, BX
	0x0215 00533 (/home/frank/arm-128base/src/crypto/des/block.go:245)	ORQ	DI, BX
	0x0218 00536 (/home/frank/arm-128base/src/crypto/des/block.go:247)	MOVQ	SI, DI
	0x021b 00539 (/home/frank/arm-128base/src/crypto/des/block.go:247)	SHLQ	$24, SI
	0x021f 00543 (/home/frank/arm-128base/src/crypto/des/block.go:247)	MOVQ	$71776119061217280, R10
	0x0229 00553 (/home/frank/arm-128base/src/crypto/des/block.go:247)	ANDQ	R10, SI
	0x022c 00556 (/home/frank/arm-128base/src/crypto/des/block.go:246)	ORQ	BX, SI
	0x022f 00559 (/home/frank/arm-128base/src/crypto/des/block.go:248)	SHLQ	$20, DI
	0x0233 00563 (/home/frank/arm-128base/src/crypto/des/block.go:248)	MOVQ	$-72057594037927936, BX
	0x023d 00573 (/home/frank/arm-128base/src/crypto/des/block.go:248)	ANDQ	BX, DI
	0x0240 00576 (/home/frank/arm-128base/src/crypto/des/block.go:247)	ORQ	SI, DI
	0x0243 00579 (<unknown line number>)	NOP
	0x0243 00579 (/home/frank/arm-128base/src/crypto/des/block.go:233)	MOVQ	DI, (AX)(DX*8)
	0x0247 00583 (/home/frank/arm-128base/src/crypto/des/block.go:229)	INCQ	DX
	0x024a 00586 (/home/frank/arm-128base/src/crypto/des/block.go:229)	CMPQ	DX, $16
	0x024e 00590 (/home/frank/arm-128base/src/crypto/des/block.go:229)	JGE	677
	0x0250 00592 (/home/frank/arm-128base/src/crypto/des/block.go:231)	MOVL	crypto/des..autotmp_46+80(SP)(DX*4), BX
	0x0254 00596 (/home/frank/arm-128base/src/crypto/des/block.go:231)	SHLQ	$28, BX
	0x0258 00600 (/home/frank/arm-128base/src/crypto/des/block.go:231)	MOVL	crypto/des..autotmp_47+16(SP)(DX*4), SI
	0x025c 00604 (/home/frank/arm-128base/src/crypto/des/block.go:231)	ORQ	SI, BX
	0x025f 00607 (<unknown line number>)	NOP
	0x025f 00607 (/home/frank/arm-128base/src/crypto/des/block.go:78)	XORL	SI, SI
	0x0261 00609 (/home/frank/arm-128base/src/crypto/des/block.go:78)	XORL	DI, DI
	0x0263 00611 (/home/frank/arm-128base/src/crypto/des/block.go:78)	JMP	663
	0x0265 00613 (/home/frank/arm-128base/src/crypto/des/block.go:78)	LEAQ	crypto/des.permutedChoice2(SB), R8
	0x026c 00620 (/home/frank/arm-128base/src/crypto/des/block.go:78)	MOVBLZX	(DI)(R8*1), CX
	0x0271 00625 (/home/frank/arm-128base/src/crypto/des/block.go:79)	MOVQ	BX, R9
	0x0274 00628 (/home/frank/arm-128base/src/crypto/des/block.go:79)	SHRQ	CX, R9
	0x0277 00631 (/home/frank/arm-128base/src/crypto/des/block.go:80)	LEAQ	-47(DI), R10
	0x027b 00635 (/home/frank/arm-128base/src/crypto/des/block.go:80)	NEGQ	R10
	0x027e 00638 (/home/frank/arm-128base/src/crypto/des/block.go:79)	CMPB	CL, $64
	0x0281 00641 (/home/frank/arm-128base/src/crypto/des/block.go:79)	SBBQ	R11, R11
	0x0284 00644 (/home/frank/arm-128base/src/crypto/des/block.go:79)	ANDQ	R11, R9
	0x0287 00647 (/home/frank/arm-128base/src/crypto/des/block.go:79)	ANDL	$1, R9
	0x028b 00651 (/home/frank/arm-128base/src/crypto/des/block.go:80)	MOVQ	R10, CX
	0x028e 00654 (/home/frank/arm-128base/src/crypto/des/block.go:80)	SHLQ	CX, R9
	0x0291 00657 (/home/frank/arm-128base/src/crypto/des/block.go:80)	ORQ	R9, SI
	0x0294 00660 (/home/frank/arm-128base/src/crypto/des/block.go:78)	INCQ	DI
	0x0297 00663 (/home/frank/arm-128base/src/crypto/des/block.go:78)	CMPQ	DI, $48
	0x029b 00667 (/home/frank/arm-128base/src/crypto/des/block.go:78)	JLT	613
	0x029d 00669 (/home/frank/arm-128base/src/crypto/des/block.go:78)	NOP
	0x02a0 00672 (/home/frank/arm-128base/src/crypto/des/block.go:78)	JMP	439
	0x02a5 00677 (/home/frank/arm-128base/src/crypto/des/block.go:235)	LEAVEQ
	0x02a6 00678 (/home/frank/arm-128base/src/crypto/des/block.go:235)	RET
	0x02a7 00679 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:121)	PCDATA	$1, $1
	0x02a7 00679 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:121)	PCDATA	$4, $1422
	0x02a7 00679 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:121)	CALL	runtime.panicBounds(SB)
	0x02ac 00684 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:121)	XCHGL	AX, AX
	0x02ad 00685 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:121)	NOP
	0x02ad 00685 (/home/frank/arm-128base/src/crypto/des/block.go:217)	PCDATA	$1, $-1
	0x02ad 00685 (/home/frank/arm-128base/src/crypto/des/block.go:217)	PCDATA	$0, $-2
	0x02ad 00685 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	AX, 8(SP)
	0x02b2 00690 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	BX, 16(SP)
	0x02b7 00695 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	CX, 24(SP)
	0x02bc 00700 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	DI, 32(SP)
	0x02c1 00705 (/home/frank/arm-128base/src/crypto/des/block.go:217)	CALL	runtime.morestack_noctxt(SB)
	0x02c6 00710 (/home/frank/arm-128base/src/crypto/des/block.go:217)	PCDATA	$0, $-1
	0x02c6 00710 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	8(SP), AX
	0x02cb 00715 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	16(SP), BX
	0x02d0 00720 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	24(SP), CX
	0x02d5 00725 (/home/frank/arm-128base/src/crypto/des/block.go:217)	MOVQ	32(SP), DI
	0x02da 00730 (/home/frank/arm-128base/src/crypto/des/block.go:217)	JMP	0
	0x0000 4c 8d 64 24 e8 4d 3b 66 10 0f 86 9e 02 00 00 55  L.d$.M;f.......U
	0x0010 48 89 e5 48 81 ec 90 00 00 00 48 89 9c 24 a8 00  H..H......H..$..
	0x0020 00 00 90 8b 15 00 00 00 00 85 d2 75 43 48 89 84  ...........uCH..
	0x0030 24 a0 00 00 00 48 89 8c 24 b0 00 00 00 48 89 9c  $....H..$....H..
	0x0040 24 a8 00 00 00 48 8d 05 00 00 00 00 48 8d 1d 00  $....H......H...
	0x0050 00 00 00 e8 00 00 00 00 48 8b 84 24 a0 00 00 00  ........H..$....
	0x0060 48 8b 8c 24 b0 00 00 00 48 8b 9c 24 a8 00 00 00  H..$....H..$....
	0x0070 90 48 83 f9 07 0f 86 2c 02 00 00 48 8b 13 90 48  .H.....,...H...H
	0x0080 0f ca 31 db 31 f6 eb 31 48 8d 3d 00 00 00 00 0f  ..1.1..1H.=.....
	0x0090 b6 0c 3b 49 89 d0 49 d3 e8 4c 8d 4b c9 49 f7 d9  ..;I..I..L.K.I..
	0x00a0 80 f9 40 4d 19 d2 4d 21 d0 41 83 e0 01 4c 89 c9  ..@M..M!.A...L..
	0x00b0 49 d3 e0 4c 09 c6 48 ff c3 48 83 fb 38 7c c9 48  I..L..H..H..8|.H
	0x00c0 8d 54 24 50 44 0f 11 3a 44 0f 11 7a 10 44 0f 11  .T$PD..:D..z.D..
	0x00d0 7a 20 44 0f 11 7a 30 48 89 f2 48 c1 ee 1c 31 db  z D..z0H..H...1.
	0x00e0 eb 52 48 8d 3d 00 00 00 00 0f b6 0c 1f 44 8d 41  .RH.=........D.A
	0x00f0 04 41 89 c9 44 89 c1 41 89 f2 41 d3 e2 c1 e6 04  .A..D..A..A.....
	0x0100 45 89 cb 41 f7 d9 44 89 c9 d3 ee 45 8d 4b e0 41  E..A..D....E.K.A
	0x0110 f7 d9 41 80 f8 20 45 19 c0 45 21 c2 41 c1 ea 04  ..A.. E..E!.A...
	0x0120 41 80 f9 20 45 19 c0 44 21 c6 44 09 d6 89 74 9c  A.. E..D!.D...t.
	0x0130 50 48 ff c3 48 83 fb 10 7c a8 48 c1 e2 04 48 8d  PH..H...|.H...H.
	0x0140 5c 24 10 44 0f 11 3b 44 0f 11 7b 10 44 0f 11 7b  \$.D..;D..{.D..{
	0x0150 20 44 0f 11 7b 30 c1 ea 04 31 db eb 4d 48 8d 35   D..{0...1..MH.5
	0x0160 00 00 00 00 0f b6 0c 33 8d 79 04 41 89 c8 89 f9  .......3.y.A....
	0x0170 41 89 d1 41 d3 e1 c1 e2 04 45 89 c2 41 f7 d8 44  A..A.....E..A..D
	0x0180 89 c1 d3 ea 45 8d 42 e0 41 f7 d8 40 80 ff 20 19  ....E.B.A..@.. .
	0x0190 ff 41 21 f9 41 c1 e9 04 41 80 f8 20 19 ff 21 fa  .A!.A...A.. ..!.
	0x01a0 44 09 ca 89 54 9c 10 48 ff c3 48 83 fb 10 7c ad  D...T..H..H...|.
	0x01b0 31 d2 e9 93 00 00 00 84 00 48 89 f3 48 c1 ee 06  1........H..H...
	0x01c0 40 0f b6 f6 48 89 df 48 c1 eb 0a 81 e3 00 ff 00  @...H..H........
	0x01d0 00 48 09 f3 48 89 fe 48 c1 ef 0e 81 e7 00 00 ff  .H..H..H........
	0x01e0 00 48 09 df 48 89 f3 48 c1 ee 12 41 b8 00 00 00  .H..H..H...A....
	0x01f0 ff 4c 21 c6 48 09 fe 0f b6 fb 48 c1 e7 20 48 09  .L!.H.....H.. H.
	0x0200 f7 48 89 de 48 c1 e3 1c 49 b9 00 00 00 00 00 ff  .H..H...I.......
	0x0210 00 00 4c 21 cb 48 09 fb 48 89 f7 48 c1 e6 18 49  ..L!.H..H..H...I
	0x0220 ba 00 00 00 00 00 00 ff 00 4c 21 d6 48 09 de 48  .........L!.H..H
	0x0230 c1 e7 14 48 bb 00 00 00 00 00 00 00 ff 48 21 df  ...H.........H!.
	0x0240 48 09 f7 48 89 3c d0 48 ff c2 48 83 fa 10 7d 55  H..H.<.H..H...}U
	0x0250 8b 5c 94 50 48 c1 e3 1c 8b 74 94 10 48 09 f3 31  .\.PH....t..H..1
	0x0260 f6 31 ff eb 32 4c 8d 05 00 00 00 00 42 0f b6 0c  .1..2L......B...
	0x0270 07 49 89 d9 49 d3 e9 4c 8d 57 d1 49 f7 da 80 f9  .I..I..L.W.I....
	0x0280 40 4d 19 db 4d 21 d9 41 83 e1 01 4c 89 d1 49 d3  @M..M!.A...L..I.
	0x0290 e1 4c 09 ce 48 ff c7 48 83 ff 30 7c c8 0f 1f 00  .L..H..H..0|....
	0x02a0 e9 12 ff ff ff c9 c3 e8 00 00 00 00 90 48 89 44  .............H.D
	0x02b0 24 08 48 89 5c 24 10 48 89 4c 24 18 48 89 7c 24  $.H.\$.H.L$.H.|$
	0x02c0 20 e8 00 00 00 00 48 8b 44 24 08 48 8b 5c 24 10   .....H.D$.H.\$.
	0x02d0 48 8b 4c 24 18 48 8b 7c 24 20 e9 21 fd ff ff     H.L$.H.|$ .!...
	rel 37+4 t=R_PCREL crypto/des.feistelBoxOnce+0
	rel 72+4 t=R_PCREL crypto/des.feistelBoxOnce+0
	rel 79+4 t=R_PCREL crypto/des.initFeistelBox·f+0
	rel 84+4 t=R_CALL sync.(*Once).doSlow+0
	rel 139+4 t=R_PCREL crypto/des.permutedChoice1+0
	rel 229+4 t=R_PCREL crypto/des.ksRotations+0
	rel 352+4 t=R_PCREL crypto/des.ksRotations+0
	rel 616+4 t=R_PCREL crypto/des.permutedChoice2+0
	rel 680+4 t=R_CALL runtime.panicBounds+0
	rel 706+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.unpack STEXT nosplit size=137 align=0x0 args=0x8 locals=0x0 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:240)	TEXT	crypto/des.unpack(SB), NOSPLIT|NOFRAME|ABIInternal, $0-8
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:240)	FUNCDATA	$0, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:240)	FUNCDATA	$1, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:240)	FUNCDATA	$5, crypto/des.unpack.arginfo1(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:240)	FUNCDATA	$6, crypto/des.unpack.argliveinfo(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:240)	PCDATA	$3, $1
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/block.go:241)	MOVQ	AX, CX
	0x0003 00003 (/home/frank/arm-128base/src/crypto/des/block.go:241)	SHRQ	$6, CX
	0x0007 00007 (/home/frank/arm-128base/src/crypto/des/block.go:241)	MOVBLZX	CL, CX
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/block.go:242)	MOVQ	AX, DX
	0x000d 00013 (/home/frank/arm-128base/src/crypto/des/block.go:242)	SHRQ	$10, DX
	0x0011 00017 (/home/frank/arm-128base/src/crypto/des/block.go:242)	ANDL	$65280, DX
	0x0017 00023 (/home/frank/arm-128base/src/crypto/des/block.go:241)	ORQ	CX, DX
	0x001a 00026 (/home/frank/arm-128base/src/crypto/des/block.go:243)	MOVQ	AX, CX
	0x001d 00029 (/home/frank/arm-128base/src/crypto/des/block.go:243)	SHRQ	$14, CX
	0x0021 00033 (/home/frank/arm-128base/src/crypto/des/block.go:243)	ANDL	$16711680, CX
	0x0027 00039 (/home/frank/arm-128base/src/crypto/des/block.go:242)	ORQ	DX, CX
	0x002a 00042 (/home/frank/arm-128base/src/crypto/des/block.go:244)	MOVQ	AX, DX
	0x002d 00045 (/home/frank/arm-128base/src/crypto/des/block.go:244)	SHRQ	$18, DX
	0x0031 00049 (/home/frank/arm-128base/src/crypto/des/block.go:244)	MOVL	$4278190080, BX
	0x0036 00054 (/home/frank/arm-128base/src/crypto/des/block.go:244)	ANDQ	DX, BX
	0x0039 00057 (/home/frank/arm-128base/src/crypto/des/block.go:243)	ORQ	CX, BX
	0x003c 00060 (/home/frank/arm-128base/src/crypto/des/block.go:245)	MOVBLZX	AL, CX
	0x003f 00063 (/home/frank/arm-128base/src/crypto/des/block.go:245)	SHLQ	$32, CX
	0x0043 00067 (/home/frank/arm-128base/src/crypto/des/block.go:244)	ORQ	BX, CX
	0x0046 00070 (/home/frank/arm-128base/src/crypto/des/block.go:246)	MOVQ	AX, DX
	0x0049 00073 (/home/frank/arm-128base/src/crypto/des/block.go:246)	SHLQ	$28, DX
	0x004d 00077 (/home/frank/arm-128base/src/crypto/des/block.go:246)	MOVQ	$280375465082880, BX
	0x0057 00087 (/home/frank/arm-128base/src/crypto/des/block.go:246)	ANDQ	DX, BX
	0x005a 00090 (/home/frank/arm-128base/src/crypto/des/block.go:245)	ORQ	CX, BX
	0x005d 00093 (/home/frank/arm-128base/src/crypto/des/block.go:247)	MOVQ	AX, CX
	0x0060 00096 (/home/frank/arm-128base/src/crypto/des/block.go:247)	SHLQ	$24, CX
	0x0064 00100 (/home/frank/arm-128base/src/crypto/des/block.go:247)	MOVQ	$71776119061217280, DX
	0x006e 00110 (/home/frank/arm-128base/src/crypto/des/block.go:247)	ANDQ	CX, DX
	0x0071 00113 (/home/frank/arm-128base/src/crypto/des/block.go:246)	ORQ	BX, DX
	0x0074 00116 (/home/frank/arm-128base/src/crypto/des/block.go:248)	SHLQ	$20, AX
	0x0078 00120 (/home/frank/arm-128base/src/crypto/des/block.go:248)	MOVQ	$-72057594037927936, CX
	0x0082 00130 (/home/frank/arm-128base/src/crypto/des/block.go:248)	ANDQ	CX, AX
	0x0085 00133 (/home/frank/arm-128base/src/crypto/des/block.go:247)	ORQ	DX, AX
	0x0088 00136 (/home/frank/arm-128base/src/crypto/des/block.go:248)	RET
	0x0000 48 89 c1 48 c1 e9 06 0f b6 c9 48 89 c2 48 c1 ea  H..H......H..H..
	0x0010 0a 81 e2 00 ff 00 00 48 09 ca 48 89 c1 48 c1 e9  .......H..H..H..
	0x0020 0e 81 e1 00 00 ff 00 48 09 d1 48 89 c2 48 c1 ea  .......H..H..H..
	0x0030 12 bb 00 00 00 ff 48 21 d3 48 09 cb 0f b6 c8 48  ......H!.H.....H
	0x0040 c1 e1 20 48 09 d9 48 89 c2 48 c1 e2 1c 48 bb 00  .. H..H..H...H..
	0x0050 00 00 00 00 ff 00 00 48 21 d3 48 09 cb 48 89 c1  .......H!.H..H..
	0x0060 48 c1 e1 18 48 ba 00 00 00 00 00 00 ff 00 48 21  H...H.........H!
	0x0070 ca 48 09 da 48 c1 e0 14 48 b9 00 00 00 00 00 00  .H..H...H.......
	0x0080 00 ff 48 21 c8 48 09 d0 c3                       ..H!.H...
crypto/des.KeySizeError.Error STEXT size=69 align=0x0 args=0x8 locals=0x30 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	TEXT	crypto/des.KeySizeError.Error(SB), ABIInternal, $48-8
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	CMPQ	SP, 16(R14)
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	PCDATA	$0, $-2
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	JLS	52
	0x0006 00006 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	PCDATA	$0, $-1
	0x0006 00006 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	PUSHQ	BP
	0x0007 00007 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	MOVQ	SP, BP
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	SUBQ	$40, SP
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	FUNCDATA	$0, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	FUNCDATA	$1, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	FUNCDATA	$5, crypto/des.KeySizeError.Error.arginfo1(SB)
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	FUNCDATA	$6, crypto/des.KeySizeError.Error.argliveinfo(SB)
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	PCDATA	$3, $1
	0x000e 00014 (<unknown line number>)	NOP
	0x000e 00014 (/home/frank/arm-128base/src/strconv/number.go:215)	XCHGL	AX, AX
	0x000f 00015 (/home/frank/arm-128base/src/internal/strconv/itoa.go:53)	MOVL	$10, BX
	0x0014 00020 (/home/frank/arm-128base/src/internal/strconv/itoa.go:53)	PCDATA	$1, $0
	0x0014 00020 (/home/frank/arm-128base/src/internal/strconv/itoa.go:53)	CALL	internal/strconv.FormatInt(SB)
	0x0019 00025 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	MOVL	$29, CX
	0x001e 00030 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	MOVQ	AX, DI
	0x0021 00033 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	MOVQ	BX, SI
	0x0024 00036 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	XORL	AX, AX
	0x0026 00038 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	LEAQ	go:string."crypto/des: invalid key size "(SB), BX
	0x002d 00045 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	CALL	runtime.concatstring2(SB)
	0x0032 00050 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	LEAVEQ
	0x0033 00051 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	RET
	0x0034 00052 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	NOP
	0x0034 00052 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	PCDATA	$1, $-1
	0x0034 00052 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	PCDATA	$0, $-2
	0x0034 00052 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	MOVQ	AX, 8(SP)
	0x0039 00057 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	CALL	runtime.morestack_noctxt(SB)
	0x003e 00062 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	PCDATA	$0, $-1
	0x003e 00062 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	MOVQ	8(SP), AX
	0x0043 00067 (/home/frank/arm-128base/src/crypto/des/cipher.go:21)	JMP	0
	0x0000 49 3b 66 10 76 2e 55 48 89 e5 48 83 ec 28 90 bb  I;f.v.UH..H..(..
	0x0010 0a 00 00 00 e8 00 00 00 00 b9 1d 00 00 00 48 89  ..............H.
	0x0020 c7 48 89 de 31 c0 48 8d 1d 00 00 00 00 e8 00 00  .H..1.H.........
	0x0030 00 00 c9 c3 48 89 44 24 08 e8 00 00 00 00 48 8b  ....H.D$......H.
	0x0040 44 24 08 eb bb                                   D$...
	rel 21+4 t=R_CALL internal/strconv.FormatInt+0
	rel 41+4 t=R_PCREL go:string."crypto/des: invalid key size "+0
	rel 46+4 t=R_CALL runtime.concatstring2+0
	rel 58+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.NewCipher STEXT size=249 align=0x0 args=0x18 locals=0x30 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	TEXT	crypto/des.NewCipher(SB), ABIInternal, $48-24
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	CMPQ	SP, 16(R14)
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	PCDATA	$0, $-2
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	JLS	208
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	PCDATA	$0, $-1
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	PUSHQ	BP
	0x000b 00011 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	MOVQ	SP, BP
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	SUBQ	$40, SP
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	FUNCDATA	$0, gclocals·Z8zdw/dq+fE82FieA9ctlQ==(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	FUNCDATA	$1, gclocals·dJUzpsp3Yf15gqnc4dLliw==(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	FUNCDATA	$5, crypto/des.NewCipher.arginfo1(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	FUNCDATA	$6, crypto/des.NewCipher.argliveinfo(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	PCDATA	$3, $1
	0x0012 00018 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	MOVQ	CX, crypto/des.key+72(SP)
	0x0017 00023 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	MOVQ	AX, crypto/des.key+56(SP)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	MOVQ	BX, crypto/des.key+64(SP)
	0x0021 00033 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	PCDATA	$3, $-1
	0x0021 00033 (<unknown line number>)	NOP
	0x0021 00033 (/home/frank/arm-128base/src/crypto/internal/fips140only/fips140only.go:18)	XCHGL	AX, AX
	0x0022 00034 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	CMPB	crypto/fips140.enabled(SB), $0
	0x0029 00041 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	JEQ	67
	0x002b 00043 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	PCDATA	$1, $0
	0x002b 00043 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	CALL	crypto/fips140.isBypassed(SB)
	0x0030 00048 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	TESTB	AL, AL
	0x0032 00050 (/home/frank/arm-128base/src/crypto/des/cipher.go:32)	JEQ	152
	0x0034 00052 (/home/frank/arm-128base/src/crypto/des/cipher.go:41)	MOVQ	crypto/des.key+56(SP), AX
	0x0039 00057 (/home/frank/arm-128base/src/crypto/des/cipher.go:41)	MOVQ	crypto/des.key+72(SP), CX
	0x003e 00062 (/home/frank/arm-128base/src/crypto/des/cipher.go:36)	MOVQ	crypto/des.key+64(SP), BX
	0x0043 00067 (/home/frank/arm-128base/src/crypto/des/cipher.go:36)	CMPQ	BX, $8
	0x0047 00071 (/home/frank/arm-128base/src/crypto/des/cipher.go:36)	JEQ	97
	0x0049 00073 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	MOVQ	BX, AX
	0x004c 00076 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	PCDATA	$1, $1
	0x004c 00076 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	CALL	runtime.convT64(SB)
	0x0051 00081 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	XORL	BX, BX
	0x0053 00083 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	LEAQ	go:itab.crypto/des.KeySizeError,error(SB), CX
	0x005a 00090 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	MOVQ	AX, DI
	0x005d 00093 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	XORL	AX, AX
	0x005f 00095 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	LEAVEQ
	0x0060 00096 (/home/frank/arm-128base/src/crypto/des/cipher.go:37)	RET
	0x0061 00097 (/home/frank/arm-128base/src/crypto/des/cipher.go:40)	LEAQ	type:crypto/des.desCipher(SB), AX
	0x0068 00104 (/home/frank/arm-128base/src/crypto/des/cipher.go:40)	PCDATA	$1, $0
	0x0068 00104 (/home/frank/arm-128base/src/crypto/des/cipher.go:40)	CALL	runtime.newobject(SB)
	0x006d 00109 (/home/frank/arm-128base/src/crypto/des/cipher.go:40)	MOVQ	AX, crypto/des.c+32(SP)
	0x0072 00114 (/home/frank/arm-128base/src/crypto/des/cipher.go:41)	MOVQ	crypto/des.key+56(SP), BX
	0x0077 00119 (/home/frank/arm-128base/src/crypto/des/cipher.go:41)	MOVQ	crypto/des.key+64(SP), CX
	0x007c 00124 (/home/frank/arm-128base/src/crypto/des/cipher.go:41)	MOVQ	crypto/des.key+72(SP), DI
	0x0081 00129 (/home/frank/arm-128base/src/crypto/des/cipher.go:41)	PCDATA	$1, $2
	0x0081 00129 (/home/frank/arm-128base/src/crypto/des/cipher.go:41)	CALL	crypto/des.(*desCipher).generateSubkeys(SB)
	0x0086 00134 (/home/frank/arm-128base/src/crypto/des/cipher.go:42)	LEAQ	go:itab.*crypto/des.desCipher,crypto/cipher.Block(SB), AX
	0x008d 00141 (/home/frank/arm-128base/src/crypto/des/cipher.go:42)	MOVQ	crypto/des.c+32(SP), BX
	0x0092 00146 (/home/frank/arm-128base/src/crypto/des/cipher.go:42)	XORL	CX, CX
	0x0094 00148 (/home/frank/arm-128base/src/crypto/des/cipher.go:42)	XORL	DI, DI
	0x0096 00150 (/home/frank/arm-128base/src/crypto/des/cipher.go:42)	LEAVEQ
	0x0097 00151 (/home/frank/arm-128base/src/crypto/des/cipher.go:42)	RET
	0x0098 00152 (<unknown line number>)	NOP
	0x0098 00152 (/home/frank/arm-128base/src/errors/errors.go:65)	MOVL	$16, AX
	0x009d 00157 (/home/frank/arm-128base/src/errors/errors.go:65)	LEAQ	type:errors.errorString(SB), BX
	0x00a4 00164 (/home/frank/arm-128base/src/errors/errors.go:65)	MOVL	$1, CX
	0x00a9 00169 (/home/frank/arm-128base/src/errors/errors.go:65)	PCDATA	$1, $1
	0x00a9 00169 (/home/frank/arm-128base/src/errors/errors.go:65)	CALL	runtime.mallocgcSmallScanNoHeaderSC2(SB)
	0x00ae 00174 (/home/frank/arm-128base/src/errors/errors.go:65)	MOVQ	$59, 8(AX)
	0x00b6 00182 (/home/frank/arm-128base/src/errors/errors.go:65)	LEAQ	go:string."crypto/des: use of DES is not allowed in FIPS 140-only mode"(SB), DX
	0x00bd 00189 (/home/frank/arm-128base/src/errors/errors.go:65)	MOVQ	DX, (AX)
	0x00c0 00192 (/home/frank/arm-128base/src/crypto/des/cipher.go:33)	XORL	BX, BX
	0x00c2 00194 (/home/frank/arm-128base/src/crypto/des/cipher.go:33)	LEAQ	go:itab.*errors.errorString,error(SB), CX
	0x00c9 00201 (/home/frank/arm-128base/src/crypto/des/cipher.go:33)	MOVQ	AX, DI
	0x00cc 00204 (/home/frank/arm-128base/src/crypto/des/cipher.go:33)	XORL	AX, AX
	0x00ce 00206 (/home/frank/arm-128base/src/crypto/des/cipher.go:33)	LEAVEQ
	0x00cf 00207 (/home/frank/arm-128base/src/crypto/des/cipher.go:33)	RET
	0x00d0 00208 (/home/frank/arm-128base/src/crypto/des/cipher.go:33)	NOP
	0x00d0 00208 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	PCDATA	$1, $-1
	0x00d0 00208 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	PCDATA	$0, $-2
	0x00d0 00208 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	MOVQ	AX, 8(SP)
	0x00d5 00213 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	MOVQ	BX, 16(SP)
	0x00da 00218 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	MOVQ	CX, 24(SP)
	0x00df 00223 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	NOP
	0x00e0 00224 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	CALL	runtime.morestack_noctxt(SB)
	0x00e5 00229 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	PCDATA	$0, $-1
	0x00e5 00229 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	MOVQ	8(SP), AX
	0x00ea 00234 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	MOVQ	16(SP), BX
	0x00ef 00239 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	MOVQ	24(SP), CX
	0x00f4 00244 (/home/frank/arm-128base/src/crypto/des/cipher.go:31)	JMP	0
	0x0000 49 3b 66 10 0f 86 c6 00 00 00 55 48 89 e5 48 83  I;f.......UH..H.
	0x0010 ec 28 48 89 4c 24 48 48 89 44 24 38 48 89 5c 24  .(H.L$HH.D$8H.\$
	0x0020 40 90 80 3d 00 00 00 00 00 74 18 e8 00 00 00 00  @..=.....t......
	0x0030 84 c0 74 64 48 8b 44 24 38 48 8b 4c 24 48 48 8b  ..tdH.D$8H.L$HH.
	0x0040 5c 24 40 48 83 fb 08 74 18 48 89 d8 e8 00 00 00  \$@H...t.H......
	0x0050 00 31 db 48 8d 0d 00 00 00 00 48 89 c7 31 c0 c9  .1.H......H..1..
	0x0060 c3 48 8d 05 00 00 00 00 e8 00 00 00 00 48 89 44  .H...........H.D
	0x0070 24 20 48 8b 5c 24 38 48 8b 4c 24 40 48 8b 7c 24  $ H.\$8H.L$@H.|$
	0x0080 48 e8 00 00 00 00 48 8d 05 00 00 00 00 48 8b 5c  H.....H......H.\
	0x0090 24 20 31 c9 31 ff c9 c3 b8 10 00 00 00 48 8d 1d  $ 1.1........H..
	0x00a0 00 00 00 00 b9 01 00 00 00 e8 00 00 00 00 48 c7  ..............H.
	0x00b0 40 08 3b 00 00 00 48 8d 15 00 00 00 00 48 89 10  @.;...H......H..
	0x00c0 31 db 48 8d 0d 00 00 00 00 48 89 c7 31 c0 c9 c3  1.H......H..1...
	0x00d0 48 89 44 24 08 48 89 5c 24 10 48 89 4c 24 18 90  H.D$.H.\$.H.L$..
	0x00e0 e8 00 00 00 00 48 8b 44 24 08 48 8b 5c 24 10 48  .....H.D$.H.\$.H
	0x00f0 8b 4c 24 18 e9 07 ff ff ff                       .L$......
	rel 3+0 t=R_USEIFACE type:*errors.errorString+0
	rel 3+0 t=R_USEIFACE type:crypto/des.KeySizeError+0
	rel 3+0 t=R_USEIFACE type:*crypto/des.desCipher+0
	rel 36+4 t=R_PCREL crypto/fips140.enabled+-1
	rel 44+4 t=R_CALL crypto/fips140.isBypassed+0
	rel 77+4 t=R_CALL runtime.convT64+0
	rel 86+4 t=R_PCREL go:itab.crypto/des.KeySizeError,error+0
	rel 100+4 t=R_PCREL type:crypto/des.desCipher+0
	rel 105+4 t=R_CALL runtime.newobject+0
	rel 130+4 t=R_CALL crypto/des.(*desCipher).generateSubkeys+0
	rel 137+4 t=R_PCREL go:itab.*crypto/des.desCipher,crypto/cipher.Block+0
	rel 160+4 t=R_PCREL type:errors.errorString+0
	rel 170+4 t=R_CALL runtime.mallocgcSmallScanNoHeaderSC2+0
	rel 185+4 t=R_PCREL go:string."crypto/des: use of DES is not allowed in FIPS 140-only mode"+0
	rel 197+4 t=R_PCREL go:itab.*errors.errorString,error+0
	rel 225+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.(*desCipher).BlockSize STEXT nosplit size=6 align=0x0 args=0x8 locals=0x0 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:45)	TEXT	crypto/des.(*desCipher).BlockSize(SB), NOSPLIT|NOFRAME|ABIInternal, $0-8
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:45)	FUNCDATA	$0, gclocals·PqVQJggGJ2Sanl8iJuPOkQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:45)	FUNCDATA	$1, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:45)	FUNCDATA	$5, crypto/des.(*desCipher).BlockSize.arginfo1(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:45)	FUNCDATA	$6, crypto/des.(*desCipher).BlockSize.argliveinfo(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:45)	PCDATA	$3, $1
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:45)	MOVL	$8, AX
	0x0005 00005 (/home/frank/arm-128base/src/crypto/des/cipher.go:45)	RET
	0x0000 b8 08 00 00 00 c3                                ......
crypto/des.(*desCipher).Encrypt STEXT size=269 align=0x0 args=0x38 locals=0x58 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	TEXT	crypto/des.(*desCipher).Encrypt(SB), ABIInternal, $88-56
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	CMPQ	SP, 16(R14)
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	PCDATA	$0, $-2
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	JLS	185
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	PCDATA	$0, $-1
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	PUSHQ	BP
	0x000b 00011 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	SP, BP
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	SUBQ	$80, SP
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	BX, crypto/des.dst+104(FP)
	0x0017 00023 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	SI, crypto/des.src+128(FP)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	FUNCDATA	$0, gclocals·rZc36ENI8UsaczlPbgMejw==(SB)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	FUNCDATA	$1, gclocals·J26BEvPExEQhJvjp9E8Whg==(SB)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	FUNCDATA	$5, crypto/des.(*desCipher).Encrypt.arginfo1(SB)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	FUNCDATA	$6, crypto/des.(*desCipher).Encrypt.argliveinfo(SB)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	PCDATA	$3, $1
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	NOP
	0x0020 00032 (/home/frank/arm-128base/src/crypto/des/cipher.go:48)	CMPQ	R8, $8
	0x0024 00036 (/home/frank/arm-128base/src/crypto/des/cipher.go:48)	JLT	165
	0x0026 00038 (/home/frank/arm-128base/src/crypto/des/cipher.go:51)	CMPQ	CX, $8
	0x002a 00042 (/home/frank/arm-128base/src/crypto/des/cipher.go:51)	JLT	142
	0x002c 00044 (<unknown line number>)	NOP
	0x002c 00044 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:26)	CMPQ	SI, BX
	0x002f 00047 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:26)	JEQ	85
	0x0031 00049 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	MOVQ	BX, DX
	0x0034 00052 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	LEAQ	7(SI), R12
	0x0038 00056 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:29)	XCHGL	AX, AX
	0x0039 00057 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	CMPQ	R12, BX
	0x003c 00060 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	JCS	82
	0x003e 00062 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	MOVQ	SI, R12
	0x0041 00065 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	LEAQ	7(DX), R13
	0x0045 00069 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	CMPQ	R13, SI
	0x0048 00072 (/home/frank/arm-128base/src/crypto/des/cipher.go:54)	JCC	123
	0x004a 00074 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	DX, BX
	0x004d 00077 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	R12, SI
	0x0050 00080 (/home/frank/arm-128base/src/crypto/des/cipher.go:54)	JMP	85
	0x0052 00082 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	DX, BX
	0x0055 00085 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	TESTB	AL, (AX)
	0x0057 00087 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVB	$0, (SP)
	0x005b 00091 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	R8, R10
	0x005e 00094 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	R9, R11
	0x0061 00097 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	DI, R8
	0x0064 00100 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	SI, R9
	0x0067 00103 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	BX, DI
	0x006a 00106 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVQ	CX, SI
	0x006d 00109 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVL	$16, BX
	0x0072 00114 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	MOVL	BX, CX
	0x0074 00116 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	PCDATA	$1, $1
	0x0074 00116 (/home/frank/arm-128base/src/crypto/des/cipher.go:57)	CALL	crypto/des.cryptBlock(SB)
	0x0079 00121 (/home/frank/arm-128base/src/crypto/des/cipher.go:58)	LEAVEQ
	0x007a 00122 (/home/frank/arm-128base/src/crypto/des/cipher.go:58)	RET
	0x007b 00123 (/home/frank/arm-128base/src/crypto/des/cipher.go:55)	LEAQ	type:string(SB), AX
	0x0082 00130 (/home/frank/arm-128base/src/crypto/des/cipher.go:55)	LEAQ	crypto/des..stmp_2(SB), BX
	0x0089 00137 (/home/frank/arm-128base/src/crypto/des/cipher.go:55)	CALL	runtime.gopanic(SB)
	0x008e 00142 (/home/frank/arm-128base/src/crypto/des/cipher.go:52)	LEAQ	type:string(SB), AX
	0x0095 00149 (/home/frank/arm-128base/src/crypto/des/cipher.go:52)	LEAQ	crypto/des..stmp_1(SB), BX
	0x009c 00156 (/home/frank/arm-128base/src/crypto/des/cipher.go:52)	NOP
	0x00a0 00160 (/home/frank/arm-128base/src/crypto/des/cipher.go:52)	CALL	runtime.gopanic(SB)
	0x00a5 00165 (/home/frank/arm-128base/src/crypto/des/cipher.go:49)	LEAQ	type:string(SB), AX
	0x00ac 00172 (/home/frank/arm-128base/src/crypto/des/cipher.go:49)	LEAQ	crypto/des..stmp_0(SB), BX
	0x00b3 00179 (/home/frank/arm-128base/src/crypto/des/cipher.go:49)	CALL	runtime.gopanic(SB)
	0x00b8 00184 (/home/frank/arm-128base/src/crypto/des/cipher.go:49)	XCHGL	AX, AX
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/cipher.go:49)	NOP
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	PCDATA	$1, $-1
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	PCDATA	$0, $-2
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	AX, 8(SP)
	0x00be 00190 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	BX, 16(SP)
	0x00c3 00195 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	CX, 24(SP)
	0x00c8 00200 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	DI, 32(SP)
	0x00cd 00205 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	SI, 40(SP)
	0x00d2 00210 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	R8, 48(SP)
	0x00d7 00215 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	R9, 56(SP)
	0x00dc 00220 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	NOP
	0x00e0 00224 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	CALL	runtime.morestack_noctxt(SB)
	0x00e5 00229 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	PCDATA	$0, $-1
	0x00e5 00229 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	8(SP), AX
	0x00ea 00234 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	16(SP), BX
	0x00ef 00239 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	24(SP), CX
	0x00f4 00244 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	32(SP), DI
	0x00f9 00249 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	40(SP), SI
	0x00fe 00254 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	48(SP), R8
	0x0103 00259 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	MOVQ	56(SP), R9
	0x0108 00264 (/home/frank/arm-128base/src/crypto/des/cipher.go:47)	JMP	0
	0x0000 49 3b 66 10 0f 86 af 00 00 00 55 48 89 e5 48 83  I;f.......UH..H.
	0x0010 ec 50 48 89 5c 24 68 48 89 b4 24 80 00 00 00 90  .PH.\$hH..$.....
	0x0020 49 83 f8 08 7c 7f 48 83 f9 08 7c 62 48 39 de 74  I...|.H...|bH9.t
	0x0030 24 48 89 da 4c 8d 66 07 90 49 39 dc 72 14 49 89  $H..L.f..I9.r.I.
	0x0040 f4 4c 8d 6a 07 49 39 f5 73 31 48 89 d3 4c 89 e6  .L.j.I9.s1H..L..
	0x0050 eb 03 48 89 d3 84 00 c6 04 24 00 4d 89 c2 4d 89  ..H......$.M..M.
	0x0060 cb 49 89 f8 49 89 f1 48 89 df 48 89 ce bb 10 00  .I..I..H..H.....
	0x0070 00 00 89 d9 e8 00 00 00 00 c9 c3 48 8d 05 00 00  ...........H....
	0x0080 00 00 48 8d 1d 00 00 00 00 e8 00 00 00 00 48 8d  ..H...........H.
	0x0090 05 00 00 00 00 48 8d 1d 00 00 00 00 0f 1f 40 00  .....H........@.
	0x00a0 e8 00 00 00 00 48 8d 05 00 00 00 00 48 8d 1d 00  .....H......H...
	0x00b0 00 00 00 e8 00 00 00 00 90 48 89 44 24 08 48 89  .........H.D$.H.
	0x00c0 5c 24 10 48 89 4c 24 18 48 89 7c 24 20 48 89 74  \$.H.L$.H.|$ H.t
	0x00d0 24 28 4c 89 44 24 30 4c 89 4c 24 38 0f 1f 40 00  $(L.D$0L.L$8..@.
	0x00e0 e8 00 00 00 00 48 8b 44 24 08 48 8b 5c 24 10 48  .....H.D$.H.\$.H
	0x00f0 8b 4c 24 18 48 8b 7c 24 20 48 8b 74 24 28 4c 8b  .L$.H.|$ H.t$(L.
	0x0100 44 24 30 4c 8b 4c 24 38 e9 f3 fe ff ff           D$0L.L$8.....
	rel 3+0 t=R_USEIFACE type:string+0
	rel 3+0 t=R_USEIFACE type:string+0
	rel 3+0 t=R_USEIFACE type:string+0
	rel 117+4 t=R_CALL crypto/des.cryptBlock+0
	rel 126+4 t=R_PCREL type:string+0
	rel 133+4 t=R_PCREL crypto/des..stmp_2+0
	rel 138+4 t=R_CALL runtime.gopanic+0
	rel 145+4 t=R_PCREL type:string+0
	rel 152+4 t=R_PCREL crypto/des..stmp_1+0
	rel 161+4 t=R_CALL runtime.gopanic+0
	rel 168+4 t=R_PCREL type:string+0
	rel 175+4 t=R_PCREL crypto/des..stmp_0+0
	rel 180+4 t=R_CALL runtime.gopanic+0
	rel 225+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.(*desCipher).Decrypt STEXT size=269 align=0x0 args=0x38 locals=0x58 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	TEXT	crypto/des.(*desCipher).Decrypt(SB), ABIInternal, $88-56
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	CMPQ	SP, 16(R14)
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	PCDATA	$0, $-2
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	JLS	185
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	PCDATA	$0, $-1
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	PUSHQ	BP
	0x000b 00011 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	SP, BP
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	SUBQ	$80, SP
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	BX, crypto/des.dst+104(FP)
	0x0017 00023 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	SI, crypto/des.src+128(FP)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	FUNCDATA	$0, gclocals·rZc36ENI8UsaczlPbgMejw==(SB)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	FUNCDATA	$1, gclocals·J26BEvPExEQhJvjp9E8Whg==(SB)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	FUNCDATA	$5, crypto/des.(*desCipher).Decrypt.arginfo1(SB)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	FUNCDATA	$6, crypto/des.(*desCipher).Decrypt.argliveinfo(SB)
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	PCDATA	$3, $1
	0x001f 00031 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	NOP
	0x0020 00032 (/home/frank/arm-128base/src/crypto/des/cipher.go:61)	CMPQ	R8, $8
	0x0024 00036 (/home/frank/arm-128base/src/crypto/des/cipher.go:61)	JLT	165
	0x0026 00038 (/home/frank/arm-128base/src/crypto/des/cipher.go:64)	CMPQ	CX, $8
	0x002a 00042 (/home/frank/arm-128base/src/crypto/des/cipher.go:64)	JLT	142
	0x002c 00044 (<unknown line number>)	NOP
	0x002c 00044 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:26)	CMPQ	SI, BX
	0x002f 00047 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:26)	JEQ	85
	0x0031 00049 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	MOVQ	BX, DX
	0x0034 00052 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	LEAQ	7(SI), R12
	0x0038 00056 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:29)	XCHGL	AX, AX
	0x0039 00057 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	CMPQ	R12, BX
	0x003c 00060 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	JCS	82
	0x003e 00062 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	MOVQ	SI, R12
	0x0041 00065 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	LEAQ	7(DX), R13
	0x0045 00069 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	CMPQ	R13, SI
	0x0048 00072 (/home/frank/arm-128base/src/crypto/des/cipher.go:67)	JCC	123
	0x004a 00074 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	DX, BX
	0x004d 00077 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	R12, SI
	0x0050 00080 (/home/frank/arm-128base/src/crypto/des/cipher.go:67)	JMP	85
	0x0052 00082 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	DX, BX
	0x0055 00085 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	TESTB	AL, (AX)
	0x0057 00087 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVB	$1, (SP)
	0x005b 00091 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	R8, R10
	0x005e 00094 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	R9, R11
	0x0061 00097 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	DI, R8
	0x0064 00100 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	SI, R9
	0x0067 00103 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	BX, DI
	0x006a 00106 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVQ	CX, SI
	0x006d 00109 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVL	$16, BX
	0x0072 00114 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	MOVL	BX, CX
	0x0074 00116 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	PCDATA	$1, $1
	0x0074 00116 (/home/frank/arm-128base/src/crypto/des/cipher.go:70)	CALL	crypto/des.cryptBlock(SB)
	0x0079 00121 (/home/frank/arm-128base/src/crypto/des/cipher.go:71)	LEAVEQ
	0x007a 00122 (/home/frank/arm-128base/src/crypto/des/cipher.go:71)	RET
	0x007b 00123 (/home/frank/arm-128base/src/crypto/des/cipher.go:68)	LEAQ	type:string(SB), AX
	0x0082 00130 (/home/frank/arm-128base/src/crypto/des/cipher.go:68)	LEAQ	crypto/des..stmp_5(SB), BX
	0x0089 00137 (/home/frank/arm-128base/src/crypto/des/cipher.go:68)	CALL	runtime.gopanic(SB)
	0x008e 00142 (/home/frank/arm-128base/src/crypto/des/cipher.go:65)	LEAQ	type:string(SB), AX
	0x0095 00149 (/home/frank/arm-128base/src/crypto/des/cipher.go:65)	LEAQ	crypto/des..stmp_4(SB), BX
	0x009c 00156 (/home/frank/arm-128base/src/crypto/des/cipher.go:65)	NOP
	0x00a0 00160 (/home/frank/arm-128base/src/crypto/des/cipher.go:65)	CALL	runtime.gopanic(SB)
	0x00a5 00165 (/home/frank/arm-128base/src/crypto/des/cipher.go:62)	LEAQ	type:string(SB), AX
	0x00ac 00172 (/home/frank/arm-128base/src/crypto/des/cipher.go:62)	LEAQ	crypto/des..stmp_3(SB), BX
	0x00b3 00179 (/home/frank/arm-128base/src/crypto/des/cipher.go:62)	CALL	runtime.gopanic(SB)
	0x00b8 00184 (/home/frank/arm-128base/src/crypto/des/cipher.go:62)	XCHGL	AX, AX
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/cipher.go:62)	NOP
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	PCDATA	$1, $-1
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	PCDATA	$0, $-2
	0x00b9 00185 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	AX, 8(SP)
	0x00be 00190 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	BX, 16(SP)
	0x00c3 00195 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	CX, 24(SP)
	0x00c8 00200 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	DI, 32(SP)
	0x00cd 00205 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	SI, 40(SP)
	0x00d2 00210 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	R8, 48(SP)
	0x00d7 00215 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	R9, 56(SP)
	0x00dc 00220 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	NOP
	0x00e0 00224 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	CALL	runtime.morestack_noctxt(SB)
	0x00e5 00229 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	PCDATA	$0, $-1
	0x00e5 00229 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	8(SP), AX
	0x00ea 00234 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	16(SP), BX
	0x00ef 00239 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	24(SP), CX
	0x00f4 00244 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	32(SP), DI
	0x00f9 00249 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	40(SP), SI
	0x00fe 00254 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	48(SP), R8
	0x0103 00259 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	MOVQ	56(SP), R9
	0x0108 00264 (/home/frank/arm-128base/src/crypto/des/cipher.go:60)	JMP	0
	0x0000 49 3b 66 10 0f 86 af 00 00 00 55 48 89 e5 48 83  I;f.......UH..H.
	0x0010 ec 50 48 89 5c 24 68 48 89 b4 24 80 00 00 00 90  .PH.\$hH..$.....
	0x0020 49 83 f8 08 7c 7f 48 83 f9 08 7c 62 48 39 de 74  I...|.H...|bH9.t
	0x0030 24 48 89 da 4c 8d 66 07 90 49 39 dc 72 14 49 89  $H..L.f..I9.r.I.
	0x0040 f4 4c 8d 6a 07 49 39 f5 73 31 48 89 d3 4c 89 e6  .L.j.I9.s1H..L..
	0x0050 eb 03 48 89 d3 84 00 c6 04 24 01 4d 89 c2 4d 89  ..H......$.M..M.
	0x0060 cb 49 89 f8 49 89 f1 48 89 df 48 89 ce bb 10 00  .I..I..H..H.....
	0x0070 00 00 89 d9 e8 00 00 00 00 c9 c3 48 8d 05 00 00  ...........H....
	0x0080 00 00 48 8d 1d 00 00 00 00 e8 00 00 00 00 48 8d  ..H...........H.
	0x0090 05 00 00 00 00 48 8d 1d 00 00 00 00 0f 1f 40 00  .....H........@.
	0x00a0 e8 00 00 00 00 48 8d 05 00 00 00 00 48 8d 1d 00  .....H......H...
	0x00b0 00 00 00 e8 00 00 00 00 90 48 89 44 24 08 48 89  .........H.D$.H.
	0x00c0 5c 24 10 48 89 4c 24 18 48 89 7c 24 20 48 89 74  \$.H.L$.H.|$ H.t
	0x00d0 24 28 4c 89 44 24 30 4c 89 4c 24 38 0f 1f 40 00  $(L.D$0L.L$8..@.
	0x00e0 e8 00 00 00 00 48 8b 44 24 08 48 8b 5c 24 10 48  .....H.D$.H.\$.H
	0x00f0 8b 4c 24 18 48 8b 7c 24 20 48 8b 74 24 28 4c 8b  .L$.H.|$ H.t$(L.
	0x0100 44 24 30 4c 8b 4c 24 38 e9 f3 fe ff ff           D$0L.L$8.....
	rel 3+0 t=R_USEIFACE type:string+0
	rel 3+0 t=R_USEIFACE type:string+0
	rel 3+0 t=R_USEIFACE type:string+0
	rel 117+4 t=R_CALL crypto/des.cryptBlock+0
	rel 126+4 t=R_PCREL type:string+0
	rel 133+4 t=R_PCREL crypto/des..stmp_5+0
	rel 138+4 t=R_CALL runtime.gopanic+0
	rel 145+4 t=R_PCREL type:string+0
	rel 152+4 t=R_PCREL crypto/des..stmp_4+0
	rel 161+4 t=R_CALL runtime.gopanic+0
	rel 168+4 t=R_PCREL type:string+0
	rel 175+4 t=R_PCREL crypto/des..stmp_3+0
	rel 180+4 t=R_CALL runtime.gopanic+0
	rel 225+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.NewTripleDESCipher STEXT size=329 align=0x0 args=0x18 locals=0x30 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	TEXT	crypto/des.NewTripleDESCipher(SB), ABIInternal, $48-24
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	CMPQ	SP, 16(R14)
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	PCDATA	$0, $-2
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	JLS	289
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	PCDATA	$0, $-1
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	PUSHQ	BP
	0x000b 00011 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	MOVQ	SP, BP
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	SUBQ	$40, SP
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	FUNCDATA	$0, gclocals·jqXdKVKXqQEE0nK8rJ96oA==(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	FUNCDATA	$1, gclocals·yjRPzIZTs1w23N5U7zM9ew==(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	FUNCDATA	$5, crypto/des.NewTripleDESCipher.arginfo1(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	FUNCDATA	$6, crypto/des.NewTripleDESCipher.argliveinfo(SB)
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	PCDATA	$3, $1
	0x0012 00018 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	MOVQ	CX, crypto/des.key+72(SP)
	0x0017 00023 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	MOVQ	AX, crypto/des.key+56(SP)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	PCDATA	$3, $2
	0x001c 00028 (<unknown line number>)	NOP
	0x001c 00028 (/home/frank/arm-128base/src/crypto/internal/fips140only/fips140only.go:18)	XCHGL	AX, AX
	0x001d 00029 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	CMPB	crypto/fips140.enabled(SB), $0
	0x0024 00036 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	JEQ	71
	0x0026 00038 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	MOVQ	BX, crypto/des.key+64(SP)
	0x002b 00043 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	PCDATA	$3, $-1
	0x002b 00043 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	PCDATA	$1, $0
	0x002b 00043 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	CALL	crypto/fips140.isBypassed(SB)
	0x0030 00048 (/home/frank/arm-128base/src/crypto/fips140/enforcement.go:37)	TESTB	AL, AL
	0x0032 00050 (/home/frank/arm-128base/src/crypto/des/cipher.go:80)	JEQ	232
	0x0038 00056 (/home/frank/arm-128base/src/crypto/des/cipher.go:89)	MOVQ	crypto/des.key+56(SP), AX
	0x003d 00061 (/home/frank/arm-128base/src/crypto/des/cipher.go:89)	MOVQ	crypto/des.key+72(SP), CX
	0x0042 00066 (/home/frank/arm-128base/src/crypto/des/cipher.go:84)	MOVQ	crypto/des.key+64(SP), BX
	0x0047 00071 (/home/frank/arm-128base/src/crypto/des/cipher.go:80)	PCDATA	$3, $2
	0x0047 00071 (/home/frank/arm-128base/src/crypto/des/cipher.go:84)	CMPQ	BX, $24
	0x004b 00075 (/home/frank/arm-128base/src/crypto/des/cipher.go:84)	JEQ	101
	0x004d 00077 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	MOVQ	BX, AX
	0x0050 00080 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	PCDATA	$1, $1
	0x0050 00080 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	CALL	runtime.convT64(SB)
	0x0055 00085 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	XORL	BX, BX
	0x0057 00087 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	LEAQ	go:itab.crypto/des.KeySizeError,error(SB), CX
	0x005e 00094 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	MOVQ	AX, DI
	0x0061 00097 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	XORL	AX, AX
	0x0063 00099 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	LEAVEQ
	0x0064 00100 (/home/frank/arm-128base/src/crypto/des/cipher.go:85)	RET
	0x0065 00101 (/home/frank/arm-128base/src/crypto/des/cipher.go:88)	LEAQ	type:crypto/des.tripleDESCipher(SB), AX
	0x006c 00108 (/home/frank/arm-128base/src/crypto/des/cipher.go:88)	PCDATA	$1, $0
	0x006c 00108 (/home/frank/arm-128base/src/crypto/des/cipher.go:88)	CALL	runtime.newobject(SB)
	0x0071 00113 (/home/frank/arm-128base/src/crypto/des/cipher.go:88)	MOVQ	AX, crypto/des.c+32(SP)
	0x0076 00118 (/home/frank/arm-128base/src/crypto/des/cipher.go:89)	MOVQ	crypto/des.key+56(SP), BX
	0x007b 00123 (/home/frank/arm-128base/src/crypto/des/cipher.go:89)	MOVL	$8, CX
	0x0080 00128 (/home/frank/arm-128base/src/crypto/des/cipher.go:89)	MOVQ	crypto/des.key+72(SP), DI
	0x0085 00133 (/home/frank/arm-128base/src/crypto/des/cipher.go:89)	PCDATA	$1, $2
	0x0085 00133 (/home/frank/arm-128base/src/crypto/des/cipher.go:89)	CALL	crypto/des.(*desCipher).generateSubkeys(SB)
	0x008a 00138 (/home/frank/arm-128base/src/crypto/des/cipher.go:90)	MOVQ	crypto/des.c+32(SP), AX
	0x008f 00143 (/home/frank/arm-128base/src/crypto/des/cipher.go:90)	SUBQ	$-128, AX
	0x0093 00147 (/home/frank/arm-128base/src/crypto/des/cipher.go:90)	MOVQ	crypto/des.key+56(SP), BX
	0x0098 00152 (/home/frank/arm-128base/src/crypto/des/cipher.go:90)	ADDQ	$8, BX
	0x009c 00156 (/home/frank/arm-128base/src/crypto/des/cipher.go:90)	MOVQ	crypto/des.key+72(SP), DI
	0x00a1 00161 (/home/frank/arm-128base/src/crypto/des/cipher.go:90)	ADDQ	$-8, DI
	0x00a5 00165 (/home/frank/arm-128base/src/crypto/des/cipher.go:90)	MOVL	$8, CX
	0x00aa 00170 (/home/frank/arm-128base/src/crypto/des/cipher.go:90)	CALL	crypto/des.(*desCipher).generateSubkeys(SB)
	0x00af 00175 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	MOVQ	crypto/des.c+32(SP), AX
	0x00b4 00180 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	ADDQ	$256, AX
	0x00ba 00186 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	MOVQ	crypto/des.key+56(SP), BX
	0x00bf 00191 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	ADDQ	$16, BX
	0x00c3 00195 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	MOVQ	crypto/des.key+72(SP), DI
	0x00c8 00200 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	ADDQ	$-16, DI
	0x00cc 00204 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	MOVL	$8, CX
	0x00d1 00209 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	PCDATA	$1, $3
	0x00d1 00209 (/home/frank/arm-128base/src/crypto/des/cipher.go:91)	CALL	crypto/des.(*desCipher).generateSubkeys(SB)
	0x00d6 00214 (/home/frank/arm-128base/src/crypto/des/cipher.go:92)	LEAQ	go:itab.*crypto/des.tripleDESCipher,crypto/cipher.Block(SB), AX
	0x00dd 00221 (/home/frank/arm-128base/src/crypto/des/cipher.go:92)	MOVQ	crypto/des.c+32(SP), BX
	0x00e2 00226 (/home/frank/arm-128base/src/crypto/des/cipher.go:92)	XORL	CX, CX
	0x00e4 00228 (/home/frank/arm-128base/src/crypto/des/cipher.go:92)	XORL	DI, DI
	0x00e6 00230 (/home/frank/arm-128base/src/crypto/des/cipher.go:92)	LEAVEQ
	0x00e7 00231 (/home/frank/arm-128base/src/crypto/des/cipher.go:92)	RET
	0x00e8 00232 (/home/frank/arm-128base/src/crypto/des/cipher.go:92)	PCDATA	$3, $-1
	0x00e8 00232 (<unknown line number>)	NOP
	0x00e8 00232 (/home/frank/arm-128base/src/errors/errors.go:65)	MOVL	$16, AX
	0x00ed 00237 (/home/frank/arm-128base/src/errors/errors.go:65)	LEAQ	type:errors.errorString(SB), BX
	0x00f4 00244 (/home/frank/arm-128base/src/errors/errors.go:65)	MOVL	$1, CX
	0x00f9 00249 (/home/frank/arm-128base/src/errors/errors.go:65)	PCDATA	$1, $1
	0x00f9 00249 (/home/frank/arm-128base/src/errors/errors.go:65)	CALL	runtime.mallocgcSmallScanNoHeaderSC2(SB)
	0x00fe 00254 (/home/frank/arm-128base/src/errors/errors.go:65)	MOVQ	$65, 8(AX)
	0x0106 00262 (/home/frank/arm-128base/src/errors/errors.go:65)	LEAQ	go:string."crypto/des: use of TripleDES is not allowed in FIPS 140-only mode"(SB), DX
	0x010d 00269 (/home/frank/arm-128base/src/errors/errors.go:65)	MOVQ	DX, (AX)
	0x0110 00272 (/home/frank/arm-128base/src/crypto/des/cipher.go:81)	XORL	BX, BX
	0x0112 00274 (/home/frank/arm-128base/src/crypto/des/cipher.go:81)	LEAQ	go:itab.*errors.errorString,error(SB), CX
	0x0119 00281 (/home/frank/arm-128base/src/crypto/des/cipher.go:81)	MOVQ	AX, DI
	0x011c 00284 (/home/frank/arm-128base/src/crypto/des/cipher.go:81)	XORL	AX, AX
	0x011e 00286 (/home/frank/arm-128base/src/crypto/des/cipher.go:81)	LEAVEQ
	0x011f 00287 (/home/frank/arm-128base/src/crypto/des/cipher.go:81)	NOP
	0x0120 00288 (/home/frank/arm-128base/src/crypto/des/cipher.go:81)	RET
	0x0121 00289 (/home/frank/arm-128base/src/crypto/des/cipher.go:81)	NOP
	0x0121 00289 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	PCDATA	$1, $-1
	0x0121 00289 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	PCDATA	$0, $-2
	0x0121 00289 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	MOVQ	AX, 8(SP)
	0x0126 00294 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	MOVQ	BX, 16(SP)
	0x012b 00299 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	MOVQ	CX, 24(SP)
	0x0130 00304 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	CALL	runtime.morestack_noctxt(SB)
	0x0135 00309 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	PCDATA	$0, $-1
	0x0135 00309 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	MOVQ	8(SP), AX
	0x013a 00314 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	MOVQ	16(SP), BX
	0x013f 00319 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	MOVQ	24(SP), CX
	0x0144 00324 (/home/frank/arm-128base/src/crypto/des/cipher.go:79)	JMP	0
	0x0000 49 3b 66 10 0f 86 17 01 00 00 55 48 89 e5 48 83  I;f.......UH..H.
	0x0010 ec 28 48 89 4c 24 48 48 89 44 24 38 90 80 3d 00  .(H.L$HH.D$8..=.
	0x0020 00 00 00 00 74 21 48 89 5c 24 40 e8 00 00 00 00  ....t!H.\$@.....
	0x0030 84 c0 0f 84 b0 00 00 00 48 8b 44 24 38 48 8b 4c  ........H.D$8H.L
	0x0040 24 48 48 8b 5c 24 40 48 83 fb 18 74 18 48 89 d8  $HH.\$@H...t.H..
	0x0050 e8 00 00 00 00 31 db 48 8d 0d 00 00 00 00 48 89  .....1.H......H.
	0x0060 c7 31 c0 c9 c3 48 8d 05 00 00 00 00 e8 00 00 00  .1...H..........
	0x0070 00 48 89 44 24 20 48 8b 5c 24 38 b9 08 00 00 00  .H.D$ H.\$8.....
	0x0080 48 8b 7c 24 48 e8 00 00 00 00 48 8b 44 24 20 48  H.|$H.....H.D$ H
	0x0090 83 e8 80 48 8b 5c 24 38 48 83 c3 08 48 8b 7c 24  ...H.\$8H...H.|$
	0x00a0 48 48 83 c7 f8 b9 08 00 00 00 e8 00 00 00 00 48  HH.............H
	0x00b0 8b 44 24 20 48 05 00 01 00 00 48 8b 5c 24 38 48  .D$ H.....H.\$8H
	0x00c0 83 c3 10 48 8b 7c 24 48 48 83 c7 f0 b9 08 00 00  ...H.|$HH.......
	0x00d0 00 e8 00 00 00 00 48 8d 05 00 00 00 00 48 8b 5c  ......H......H.\
	0x00e0 24 20 31 c9 31 ff c9 c3 b8 10 00 00 00 48 8d 1d  $ 1.1........H..
	0x00f0 00 00 00 00 b9 01 00 00 00 e8 00 00 00 00 48 c7  ..............H.
	0x0100 40 08 41 00 00 00 48 8d 15 00 00 00 00 48 89 10  @.A...H......H..
	0x0110 31 db 48 8d 0d 00 00 00 00 48 89 c7 31 c0 c9 90  1.H......H..1...
	0x0120 c3 48 89 44 24 08 48 89 5c 24 10 48 89 4c 24 18  .H.D$.H.\$.H.L$.
	0x0130 e8 00 00 00 00 48 8b 44 24 08 48 8b 5c 24 10 48  .....H.D$.H.\$.H
	0x0140 8b 4c 24 18 e9 b7 fe ff ff                       .L$......
	rel 3+0 t=R_USEIFACE type:*errors.errorString+0
	rel 3+0 t=R_USEIFACE type:crypto/des.KeySizeError+0
	rel 3+0 t=R_USEIFACE type:*crypto/des.tripleDESCipher+0
	rel 31+4 t=R_PCREL crypto/fips140.enabled+-1
	rel 44+4 t=R_CALL crypto/fips140.isBypassed+0
	rel 81+4 t=R_CALL runtime.convT64+0
	rel 90+4 t=R_PCREL go:itab.crypto/des.KeySizeError,error+0
	rel 104+4 t=R_PCREL type:crypto/des.tripleDESCipher+0
	rel 109+4 t=R_CALL runtime.newobject+0
	rel 134+4 t=R_CALL crypto/des.(*desCipher).generateSubkeys+0
	rel 171+4 t=R_CALL crypto/des.(*desCipher).generateSubkeys+0
	rel 210+4 t=R_CALL crypto/des.(*desCipher).generateSubkeys+0
	rel 217+4 t=R_PCREL go:itab.*crypto/des.tripleDESCipher,crypto/cipher.Block+0
	rel 240+4 t=R_PCREL type:errors.errorString+0
	rel 250+4 t=R_CALL runtime.mallocgcSmallScanNoHeaderSC2+0
	rel 265+4 t=R_PCREL go:string."crypto/des: use of TripleDES is not allowed in FIPS 140-only mode"+0
	rel 277+4 t=R_PCREL go:itab.*errors.errorString,error+0
	rel 305+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.(*tripleDESCipher).BlockSize STEXT nosplit size=6 align=0x0 args=0x8 locals=0x0 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:95)	TEXT	crypto/des.(*tripleDESCipher).BlockSize(SB), NOSPLIT|NOFRAME|ABIInternal, $0-8
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:95)	FUNCDATA	$0, gclocals·PqVQJggGJ2Sanl8iJuPOkQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:95)	FUNCDATA	$1, gclocals·g5+hNtRBP6YXNjfog7aZjQ==(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:95)	FUNCDATA	$5, crypto/des.(*tripleDESCipher).BlockSize.arginfo1(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:95)	FUNCDATA	$6, crypto/des.(*tripleDESCipher).BlockSize.argliveinfo(SB)
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:95)	PCDATA	$3, $1
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:95)	MOVL	$8, AX
	0x0005 00005 (/home/frank/arm-128base/src/crypto/des/cipher.go:95)	RET
	0x0000 b8 08 00 00 00 c3                                ......
crypto/des.(*tripleDESCipher).Encrypt STEXT size=517 align=0x0 args=0x38 locals=0x38 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	TEXT	crypto/des.(*tripleDESCipher).Encrypt(SB), ABIInternal, $56-56
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	CMPQ	SP, 16(R14)
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	PCDATA	$0, $-2
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	JLS	434
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	PCDATA	$0, $-1
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	PUSHQ	BP
	0x000b 00011 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	SP, BP
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	SUBQ	$48, SP
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	BX, crypto/des.dst+72(FP)
	0x0017 00023 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	SI, crypto/des.src+96(FP)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	FUNCDATA	$0, gclocals·+CyRcNapSOQVXQOoLfXQCQ==(SB)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	FUNCDATA	$1, gclocals·D6AnjkNyRZzKYVnNXnHP7g==(SB)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	FUNCDATA	$5, crypto/des.(*tripleDESCipher).Encrypt.arginfo1(SB)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	FUNCDATA	$6, crypto/des.(*tripleDESCipher).Encrypt.argliveinfo(SB)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	PCDATA	$3, $1
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	NOP
	0x0020 00032 (/home/frank/arm-128base/src/crypto/des/cipher.go:98)	CMPQ	R8, $8
	0x0024 00036 (/home/frank/arm-128base/src/crypto/des/cipher.go:98)	JLT	414
	0x002a 00042 (/home/frank/arm-128base/src/crypto/des/cipher.go:101)	CMPQ	CX, $8
	0x002e 00046 (/home/frank/arm-128base/src/crypto/des/cipher.go:101)	JLT	395
	0x0034 00052 (/home/frank/arm-128base/src/crypto/des/cipher.go:98)	MOVQ	AX, crypto/des.c+64(SP)
	0x0039 00057 (/home/frank/arm-128base/src/crypto/des/cipher.go:98)	MOVQ	BX, crypto/des.dst+72(SP)
	0x003e 00062 (/home/frank/arm-128base/src/crypto/des/cipher.go:98)	PCDATA	$3, $2
	0x003e 00062 (<unknown line number>)	NOP
	0x003e 00062 (<unknown line number>)	NOP
	0x0040 00064 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:26)	CMPQ	BX, SI
	0x0043 00067 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:26)	JEQ	108
	0x0045 00069 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	MOVQ	BX, CX
	0x0048 00072 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	LEAQ	7(SI), DX
	0x004c 00076 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:29)	XCHGL	AX, AX
	0x004d 00077 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	CMPQ	DX, BX
	0x0050 00080 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	JCS	108
	0x0052 00082 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	MOVQ	SI, DX
	0x0055 00085 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	LEAQ	7(CX), DI
	0x0059 00089 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	NOP
	0x0060 00096 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	CMPQ	DI, SI
	0x0063 00099 (/home/frank/arm-128base/src/crypto/des/cipher.go:104)	JCC	376
	0x0069 00105 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	MOVQ	DX, SI
	0x006c 00108 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	MOVQ	(SI), AX
	0x006f 00111 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	BSWAPQ	AX
	0x0072 00114 (/home/frank/arm-128base/src/crypto/des/cipher.go:108)	XCHGL	AX, AX
	0x0073 00115 (/home/frank/arm-128base/src/crypto/des/cipher.go:109)	PCDATA	$1, $1
	0x0073 00115 (/home/frank/arm-128base/src/crypto/des/cipher.go:109)	CALL	crypto/des.permuteInitialBlock(SB)
	0x0078 00120 (/home/frank/arm-128base/src/crypto/des/cipher.go:110)	MOVQ	AX, BX
	0x007b 00123 (/home/frank/arm-128base/src/crypto/des/cipher.go:110)	SHRQ	$32, BX
	0x007f 00127 (/home/frank/arm-128base/src/crypto/des/cipher.go:112)	ROLL	$1, BX
	0x0081 00129 (/home/frank/arm-128base/src/crypto/des/cipher.go:113)	ROLL	$1, AX
	0x0083 00131 (/home/frank/arm-128base/src/crypto/des/cipher.go:113)	XORL	CX, CX
	0x0085 00133 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	JMP	192
	0x0087 00135 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	MOVQ	CX, crypto/des.i+40(SP)
	0x008c 00140 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	MOVQ	crypto/des.c+64(SP), DX
	0x0091 00145 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	TESTB	AL, (DX)
	0x0093 00147 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	MOVQ	CX, SI
	0x0096 00150 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	SHLQ	$4, CX
	0x009a 00154 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	MOVQ	(DX)(CX*1), CX
	0x009e 00158 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	LEAQ	1(SI)(SI*1), SI
	0x00a3 00163 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	MOVQ	(DX)(SI*8), DI
	0x00a7 00167 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	MOVL	AX, DX
	0x00a9 00169 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	MOVL	BX, AX
	0x00ab 00171 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	MOVL	DX, BX
	0x00ad 00173 (/home/frank/arm-128base/src/crypto/des/cipher.go:116)	CALL	crypto/des.feistel(SB)
	0x00b2 00178 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	MOVQ	crypto/des.i+40(SP), CX
	0x00b7 00183 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	INCQ	CX
	0x00ba 00186 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	MOVL	AX, DX
	0x00bc 00188 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	MOVL	BX, AX
	0x00be 00190 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	MOVL	DX, BX
	0x00c0 00192 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	CMPQ	CX, $8
	0x00c4 00196 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	JLT	135
	0x00c6 00198 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	XORL	CX, CX
	0x00c8 00200 (/home/frank/arm-128base/src/crypto/des/cipher.go:115)	JMP	262
	0x00ca 00202 (/home/frank/arm-128base/src/crypto/des/cipher.go:118)	MOVQ	CX, crypto/des.i+32(SP)
	0x00cf 00207 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	MOVQ	crypto/des.c+64(SP), DX
	0x00d4 00212 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	TESTB	AL, (DX)
	0x00d6 00214 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	LEAQ	-15(CX)(CX*1), SI
	0x00db 00219 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	NEGQ	SI
	0x00de 00222 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	MOVQ	128(DX)(SI*8), SI
	0x00e6 00230 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	LEAQ	-14(CX)(CX*1), R8
	0x00eb 00235 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	NEGQ	R8
	0x00ee 00238 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	MOVQ	128(DX)(R8*8), DI
	0x00f6 00246 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	MOVQ	SI, CX
	0x00f9 00249 (/home/frank/arm-128base/src/crypto/des/cipher.go:119)	CALL	crypto/des.feistel(SB)
	0x00fe 00254 (/home/frank/arm-128base/src/crypto/des/cipher.go:118)	MOVQ	crypto/des.i+32(SP), CX
	0x0103 00259 (/home/frank/arm-128base/src/crypto/des/cipher.go:118)	INCQ	CX
	0x0106 00262 (/home/frank/arm-128base/src/crypto/des/cipher.go:118)	CMPQ	CX, $8
	0x010a 00266 (/home/frank/arm-128base/src/crypto/des/cipher.go:118)	JLT	202
	0x010c 00268 (/home/frank/arm-128base/src/crypto/des/cipher.go:118)	XORL	CX, CX
	0x010e 00270 (/home/frank/arm-128base/src/crypto/des/cipher.go:118)	JMP	339
	0x0110 00272 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	MOVQ	CX, crypto/des.i+24(SP)
	0x0115 00277 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	MOVQ	crypto/des.c+64(SP), DX
	0x011a 00282 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	TESTB	AL, (DX)
	0x011c 00284 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	MOVQ	CX, SI
	0x011f 00287 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	SHLQ	$4, CX
	0x0123 00291 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	MOVQ	256(DX)(CX*1), CX
	0x012b 00299 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	LEAQ	1(SI)(SI*1), SI
	0x0130 00304 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	MOVQ	256(DX)(SI*8), DI
	0x0138 00312 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	MOVL	AX, DX
	0x013a 00314 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	MOVL	BX, AX
	0x013c 00316 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	MOVL	DX, BX
	0x013e 00318 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	NOP
	0x0140 00320 (/home/frank/arm-128base/src/crypto/des/cipher.go:122)	CALL	crypto/des.feistel(SB)
	0x0145 00325 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	MOVQ	crypto/des.i+24(SP), CX
	0x014a 00330 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	INCQ	CX
	0x014d 00333 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	MOVL	AX, DX
	0x014f 00335 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	MOVL	BX, AX
	0x0151 00337 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	MOVL	DX, BX
	0x0153 00339 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	CMPQ	CX, $8
	0x0157 00343 (/home/frank/arm-128base/src/crypto/des/cipher.go:121)	JLT	272
	0x0159 00345 (/home/frank/arm-128base/src/crypto/des/cipher.go:125)	ROLL	$31, BX
	0x015c 00348 (/home/frank/arm-128base/src/crypto/des/cipher.go:126)	ROLL	$31, AX
	0x015f 00351 (/home/frank/arm-128base/src/crypto/des/cipher.go:128)	SHLQ	$32, AX
	0x0163 00355 (/home/frank/arm-128base/src/crypto/des/cipher.go:128)	ORQ	BX, AX
	0x0166 00358 (/home/frank/arm-128base/src/crypto/des/cipher.go:129)	PCDATA	$1, $2
	0x0166 00358 (/home/frank/arm-128base/src/crypto/des/cipher.go:129)	CALL	crypto/des.permuteFinalBlock(SB)
	0x016b 00363 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:135)	BSWAPQ	AX
	0x016e 00366 (<unknown line number>)	NOP
	0x016e 00366 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:128)	MOVQ	crypto/des.dst+72(SP), CX
	0x0173 00371 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:128)	MOVQ	AX, (CX)
	0x0176 00374 (/home/frank/arm-128base/src/crypto/des/cipher.go:130)	LEAVEQ
	0x0177 00375 (/home/frank/arm-128base/src/crypto/des/cipher.go:130)	RET
	0x0178 00376 (/home/frank/arm-128base/src/crypto/des/cipher.go:105)	LEAQ	type:string(SB), AX
	0x017f 00383 (/home/frank/arm-128base/src/crypto/des/cipher.go:105)	LEAQ	crypto/des..stmp_8(SB), BX
	0x0186 00390 (/home/frank/arm-128base/src/crypto/des/cipher.go:105)	PCDATA	$1, $3
	0x0186 00390 (/home/frank/arm-128base/src/crypto/des/cipher.go:105)	CALL	runtime.gopanic(SB)
	0x018b 00395 (/home/frank/arm-128base/src/crypto/des/cipher.go:105)	PCDATA	$3, $1
	0x018b 00395 (/home/frank/arm-128base/src/crypto/des/cipher.go:102)	LEAQ	type:string(SB), AX
	0x0192 00402 (/home/frank/arm-128base/src/crypto/des/cipher.go:102)	LEAQ	crypto/des..stmp_7(SB), BX
	0x0199 00409 (/home/frank/arm-128base/src/crypto/des/cipher.go:102)	CALL	runtime.gopanic(SB)
	0x019e 00414 (/home/frank/arm-128base/src/crypto/des/cipher.go:99)	LEAQ	type:string(SB), AX
	0x01a5 00421 (/home/frank/arm-128base/src/crypto/des/cipher.go:99)	LEAQ	crypto/des..stmp_6(SB), BX
	0x01ac 00428 (/home/frank/arm-128base/src/crypto/des/cipher.go:99)	CALL	runtime.gopanic(SB)
	0x01b1 00433 (/home/frank/arm-128base/src/crypto/des/cipher.go:99)	XCHGL	AX, AX
	0x01b2 00434 (/home/frank/arm-128base/src/crypto/des/cipher.go:99)	NOP
	0x01b2 00434 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	PCDATA	$1, $-1
	0x01b2 00434 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	PCDATA	$0, $-2
	0x01b2 00434 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	AX, 8(SP)
	0x01b7 00439 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	BX, 16(SP)
	0x01bc 00444 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	CX, 24(SP)
	0x01c1 00449 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	DI, 32(SP)
	0x01c6 00454 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	SI, 40(SP)
	0x01cb 00459 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	R8, 48(SP)
	0x01d0 00464 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	R9, 56(SP)
	0x01d5 00469 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	CALL	runtime.morestack_noctxt(SB)
	0x01da 00474 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	PCDATA	$0, $-1
	0x01da 00474 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	8(SP), AX
	0x01df 00479 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	16(SP), BX
	0x01e4 00484 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	24(SP), CX
	0x01e9 00489 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	32(SP), DI
	0x01ee 00494 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	40(SP), SI
	0x01f3 00499 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	48(SP), R8
	0x01f8 00504 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	MOVQ	56(SP), R9
	0x01fd 00509 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	NOP
	0x0200 00512 (/home/frank/arm-128base/src/crypto/des/cipher.go:97)	JMP	0
	0x0000 49 3b 66 10 0f 86 a8 01 00 00 55 48 89 e5 48 83  I;f.......UH..H.
	0x0010 ec 30 48 89 5c 24 48 48 89 74 24 60 0f 1f 40 00  .0H.\$HH.t$`..@.
	0x0020 49 83 f8 08 0f 8c 74 01 00 00 48 83 f9 08 0f 8c  I.....t...H.....
	0x0030 57 01 00 00 48 89 44 24 40 48 89 5c 24 48 66 90  W...H.D$@H.\$Hf.
	0x0040 48 39 f3 74 27 48 89 d9 48 8d 56 07 90 48 39 da  H9.t'H..H.V..H9.
	0x0050 72 1a 48 89 f2 48 8d 79 07 0f 1f 80 00 00 00 00  r.H..H.y........
	0x0060 48 39 f7 0f 83 0f 01 00 00 48 89 d6 48 8b 06 48  H9.......H..H..H
	0x0070 0f c8 90 e8 00 00 00 00 48 89 c3 48 c1 eb 20 d1  ........H..H.. .
	0x0080 c3 d1 c0 31 c9 eb 39 48 89 4c 24 28 48 8b 54 24  ...1..9H.L$(H.T$
	0x0090 40 84 02 48 89 ce 48 c1 e1 04 48 8b 0c 0a 48 8d  @..H..H...H...H.
	0x00a0 74 36 01 48 8b 3c f2 89 c2 89 d8 89 d3 e8 00 00  t6.H.<..........
	0x00b0 00 00 48 8b 4c 24 28 48 ff c1 89 c2 89 d8 89 d3  ..H.L$(H........
	0x00c0 48 83 f9 08 7c c1 31 c9 eb 3c 48 89 4c 24 20 48  H...|.1..<H.L$ H
	0x00d0 8b 54 24 40 84 02 48 8d 74 09 f1 48 f7 de 48 8b  .T$@..H.t..H..H.
	0x00e0 b4 f2 80 00 00 00 4c 8d 44 09 f2 49 f7 d8 4a 8b  ......L.D..I..J.
	0x00f0 bc c2 80 00 00 00 48 89 f1 e8 00 00 00 00 48 8b  ......H.......H.
	0x0100 4c 24 20 48 ff c1 48 83 f9 08 7c be 31 c9 eb 43  L$ H..H...|.1..C
	0x0110 48 89 4c 24 18 48 8b 54 24 40 84 02 48 89 ce 48  H.L$.H.T$@..H..H
	0x0120 c1 e1 04 48 8b 8c 0a 00 01 00 00 48 8d 74 36 01  ...H.......H.t6.
	0x0130 48 8b bc f2 00 01 00 00 89 c2 89 d8 89 d3 66 90  H.............f.
	0x0140 e8 00 00 00 00 48 8b 4c 24 18 48 ff c1 89 c2 89  .....H.L$.H.....
	0x0150 d8 89 d3 48 83 f9 08 7c b7 c1 c3 1f c1 c0 1f 48  ...H...|.......H
	0x0160 c1 e0 20 48 09 d8 e8 00 00 00 00 48 0f c8 48 8b  .. H.......H..H.
	0x0170 4c 24 48 48 89 01 c9 c3 48 8d 05 00 00 00 00 48  L$HH....H......H
	0x0180 8d 1d 00 00 00 00 e8 00 00 00 00 48 8d 05 00 00  ...........H....
	0x0190 00 00 48 8d 1d 00 00 00 00 e8 00 00 00 00 48 8d  ..H...........H.
	0x01a0 05 00 00 00 00 48 8d 1d 00 00 00 00 e8 00 00 00  .....H..........
	0x01b0 00 90 48 89 44 24 08 48 89 5c 24 10 48 89 4c 24  ..H.D$.H.\$.H.L$
	0x01c0 18 48 89 7c 24 20 48 89 74 24 28 4c 89 44 24 30  .H.|$ H.t$(L.D$0
	0x01d0 4c 89 4c 24 38 e8 00 00 00 00 48 8b 44 24 08 48  L.L$8.....H.D$.H
	0x01e0 8b 5c 24 10 48 8b 4c 24 18 48 8b 7c 24 20 48 8b  .\$.H.L$.H.|$ H.
	0x01f0 74 24 28 4c 8b 44 24 30 4c 8b 4c 24 38 0f 1f 00  t$(L.D$0L.L$8...
	0x0200 e9 fb fd ff ff                                   .....
	rel 3+0 t=R_USEIFACE type:string+0
	rel 3+0 t=R_USEIFACE type:string+0
	rel 3+0 t=R_USEIFACE type:string+0
	rel 116+4 t=R_CALL crypto/des.permuteInitialBlock+0
	rel 174+4 t=R_CALL crypto/des.feistel+0
	rel 250+4 t=R_CALL crypto/des.feistel+0
	rel 321+4 t=R_CALL crypto/des.feistel+0
	rel 359+4 t=R_CALL crypto/des.permuteFinalBlock+0
	rel 379+4 t=R_PCREL type:string+0
	rel 386+4 t=R_PCREL crypto/des..stmp_8+0
	rel 391+4 t=R_CALL runtime.gopanic+0
	rel 398+4 t=R_PCREL type:string+0
	rel 405+4 t=R_PCREL crypto/des..stmp_7+0
	rel 410+4 t=R_CALL runtime.gopanic+0
	rel 417+4 t=R_PCREL type:string+0
	rel 424+4 t=R_PCREL crypto/des..stmp_6+0
	rel 429+4 t=R_CALL runtime.gopanic+0
	rel 470+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.(*tripleDESCipher).Decrypt STEXT size=527 align=0x0 args=0x38 locals=0x38 funcid=0x0
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	TEXT	crypto/des.(*tripleDESCipher).Decrypt(SB), ABIInternal, $56-56
	0x0000 00000 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	CMPQ	SP, 16(R14)
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	PCDATA	$0, $-2
	0x0004 00004 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	JLS	447
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	PCDATA	$0, $-1
	0x000a 00010 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	PUSHQ	BP
	0x000b 00011 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	SP, BP
	0x000e 00014 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	SUBQ	$48, SP
	0x0012 00018 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	BX, crypto/des.dst+72(FP)
	0x0017 00023 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	SI, crypto/des.src+96(FP)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	FUNCDATA	$0, gclocals·+CyRcNapSOQVXQOoLfXQCQ==(SB)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	FUNCDATA	$1, gclocals·D6AnjkNyRZzKYVnNXnHP7g==(SB)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	FUNCDATA	$5, crypto/des.(*tripleDESCipher).Decrypt.arginfo1(SB)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	FUNCDATA	$6, crypto/des.(*tripleDESCipher).Decrypt.argliveinfo(SB)
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	PCDATA	$3, $1
	0x001c 00028 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	NOP
	0x0020 00032 (/home/frank/arm-128base/src/crypto/des/cipher.go:133)	CMPQ	R8, $8
	0x0024 00036 (/home/frank/arm-128base/src/crypto/des/cipher.go:133)	JLT	427
	0x002a 00042 (/home/frank/arm-128base/src/crypto/des/cipher.go:136)	CMPQ	CX, $8
	0x002e 00046 (/home/frank/arm-128base/src/crypto/des/cipher.go:136)	JLT	408
	0x0034 00052 (/home/frank/arm-128base/src/crypto/des/cipher.go:133)	MOVQ	AX, crypto/des.c+64(SP)
	0x0039 00057 (/home/frank/arm-128base/src/crypto/des/cipher.go:133)	MOVQ	BX, crypto/des.dst+72(SP)
	0x003e 00062 (/home/frank/arm-128base/src/crypto/des/cipher.go:133)	PCDATA	$3, $2
	0x003e 00062 (<unknown line number>)	NOP
	0x003e 00062 (<unknown line number>)	NOP
	0x0040 00064 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:26)	CMPQ	BX, SI
	0x0043 00067 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:26)	JEQ	108
	0x0045 00069 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	MOVQ	BX, CX
	0x0048 00072 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	LEAQ	7(SI), DX
	0x004c 00076 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:29)	XCHGL	AX, AX
	0x004d 00077 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	CMPQ	DX, BX
	0x0050 00080 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:15)	JCS	108
	0x0052 00082 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	MOVQ	SI, DX
	0x0055 00085 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	LEAQ	7(CX), DI
	0x0059 00089 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	NOP
	0x0060 00096 (/home/frank/arm-128base/src/crypto/internal/fips140/alias/alias.go:16)	CMPQ	DI, SI
	0x0063 00099 (/home/frank/arm-128base/src/crypto/des/cipher.go:139)	JCC	389
	0x0069 00105 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	MOVQ	DX, SI
	0x006c 00108 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	MOVQ	(SI), AX
	0x006f 00111 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:123)	BSWAPQ	AX
	0x0072 00114 (/home/frank/arm-128base/src/crypto/des/cipher.go:143)	XCHGL	AX, AX
	0x0073 00115 (/home/frank/arm-128base/src/crypto/des/cipher.go:144)	PCDATA	$1, $1
	0x0073 00115 (/home/frank/arm-128base/src/crypto/des/cipher.go:144)	CALL	crypto/des.permuteInitialBlock(SB)
	0x0078 00120 (/home/frank/arm-128base/src/crypto/des/cipher.go:145)	MOVQ	AX, BX
	0x007b 00123 (/home/frank/arm-128base/src/crypto/des/cipher.go:145)	SHRQ	$32, BX
	0x007f 00127 (/home/frank/arm-128base/src/crypto/des/cipher.go:147)	ROLL	$1, BX
	0x0081 00129 (/home/frank/arm-128base/src/crypto/des/cipher.go:148)	ROLL	$1, AX
	0x0083 00131 (/home/frank/arm-128base/src/crypto/des/cipher.go:148)	XORL	CX, CX
	0x0085 00133 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	JMP	211
	0x0087 00135 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	MOVQ	CX, crypto/des.i+40(SP)
	0x008c 00140 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	MOVQ	crypto/des.c+64(SP), DX
	0x0091 00145 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	TESTB	AL, (DX)
	0x0093 00147 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	LEAQ	-15(CX)(CX*1), SI
	0x0098 00152 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	NEGQ	SI
	0x009b 00155 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	MOVQ	256(DX)(SI*8), SI
	0x00a3 00163 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	LEAQ	-14(CX)(CX*1), R8
	0x00a8 00168 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	NEGQ	R8
	0x00ab 00171 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	MOVQ	256(DX)(R8*8), DI
	0x00b3 00179 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	MOVQ	SI, CX
	0x00b6 00182 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	MOVL	AX, DX
	0x00b8 00184 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	MOVL	BX, AX
	0x00ba 00186 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	MOVL	DX, BX
	0x00bc 00188 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	NOP
	0x00c0 00192 (/home/frank/arm-128base/src/crypto/des/cipher.go:151)	CALL	crypto/des.feistel(SB)
	0x00c5 00197 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	MOVQ	crypto/des.i+40(SP), CX
	0x00ca 00202 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	INCQ	CX
	0x00cd 00205 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	MOVL	AX, DX
	0x00cf 00207 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	MOVL	BX, AX
	0x00d1 00209 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	MOVL	DX, BX
	0x00d3 00211 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	CMPQ	CX, $8
	0x00d7 00215 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	JLT	135
	0x00d9 00217 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	XORL	CX, CX
	0x00db 00219 (/home/frank/arm-128base/src/crypto/des/cipher.go:150)	JMP	274
	0x00dd 00221 (/home/frank/arm-128base/src/crypto/des/cipher.go:153)	MOVQ	CX, crypto/des.i+32(SP)
	0x00e2 00226 (/home/frank/arm-128base/src/crypto/des/cipher.go:154)	MOVQ	crypto/des.c+64(SP), DX
	0x00e7 00231 (/home/frank/arm-128base/src/crypto/des/cipher.go:154)	TESTB	AL, (DX)
	0x00e9 00233 (/home/frank/arm-128base/src/crypto/des/cipher.go:154)	MOVQ	CX, SI
	0x00ec 00236 (/home/frank/arm-128base/src/crypto/des/cipher.go:154)	SHLQ	$4, CX
	0x00f0 00240 (/home/frank/arm-128base/src/crypto/des/cipher.go:154)	MOVQ	128(DX)(CX*1), CX
	0x00f8 00248 (/home/frank/arm-128base/src/crypto/des/cipher.go:154)	LEAQ	1(SI)(SI*1), SI
	0x00fd 00253 (/home/frank/arm-128base/src/crypto/des/cipher.go:154)	MOVQ	128(DX)(SI*8), DI
	0x0105 00261 (/home/frank/arm-128base/src/crypto/des/cipher.go:154)	CALL	crypto/des.feistel(SB)
	0x010a 00266 (/home/frank/arm-128base/src/crypto/des/cipher.go:153)	MOVQ	crypto/des.i+32(SP), CX
	0x010f 00271 (/home/frank/arm-128base/src/crypto/des/cipher.go:153)	INCQ	CX
	0x0112 00274 (/home/frank/arm-128base/src/crypto/des/cipher.go:153)	CMPQ	CX, $8
	0x0116 00278 (/home/frank/arm-128base/src/crypto/des/cipher.go:153)	JLT	221
	0x0118 00280 (/home/frank/arm-128base/src/crypto/des/cipher.go:153)	XORL	CX, CX
	0x011a 00282 (/home/frank/arm-128base/src/crypto/des/cipher.go:153)	JMP	352
	0x011c 00284 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	MOVQ	CX, crypto/des.i+24(SP)
	0x0121 00289 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	MOVQ	crypto/des.c+64(SP), DX
	0x0126 00294 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	TESTB	AL, (DX)
	0x0128 00296 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	LEAQ	-15(CX)(CX*1), SI
	0x012d 00301 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	NEGQ	SI
	0x0130 00304 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	MOVQ	(DX)(SI*8), SI
	0x0134 00308 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	LEAQ	-14(CX)(CX*1), R8
	0x0139 00313 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	NEGQ	R8
	0x013c 00316 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	MOVQ	(DX)(R8*8), DI
	0x0140 00320 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	MOVQ	SI, CX
	0x0143 00323 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	MOVL	AX, DX
	0x0145 00325 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	MOVL	BX, AX
	0x0147 00327 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	MOVL	DX, BX
	0x0149 00329 (/home/frank/arm-128base/src/crypto/des/cipher.go:157)	CALL	crypto/des.feistel(SB)
	0x014e 00334 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	MOVQ	crypto/des.i+24(SP), CX
	0x0153 00339 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	INCQ	CX
	0x0156 00342 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	MOVL	AX, DX
	0x0158 00344 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	MOVL	BX, AX
	0x015a 00346 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	MOVL	DX, BX
	0x015c 00348 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	NOP
	0x0160 00352 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	CMPQ	CX, $8
	0x0164 00356 (/home/frank/arm-128base/src/crypto/des/cipher.go:156)	JLT	284
	0x0166 00358 (/home/frank/arm-128base/src/crypto/des/cipher.go:160)	ROLL	$31, BX
	0x0169 00361 (/home/frank/arm-128base/src/crypto/des/cipher.go:161)	ROLL	$31, AX
	0x016c 00364 (/home/frank/arm-128base/src/crypto/des/cipher.go:163)	SHLQ	$32, AX
	0x0170 00368 (/home/frank/arm-128base/src/crypto/des/cipher.go:163)	ORQ	BX, AX
	0x0173 00371 (/home/frank/arm-128base/src/crypto/des/cipher.go:164)	PCDATA	$1, $2
	0x0173 00371 (/home/frank/arm-128base/src/crypto/des/cipher.go:164)	CALL	crypto/des.permuteFinalBlock(SB)
	0x0178 00376 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:135)	BSWAPQ	AX
	0x017b 00379 (<unknown line number>)	NOP
	0x017b 00379 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:128)	MOVQ	crypto/des.dst+72(SP), CX
	0x0180 00384 (/home/frank/arm-128base/src/internal/byteorder/byteorder.go:128)	MOVQ	AX, (CX)
	0x0183 00387 (/home/frank/arm-128base/src/crypto/des/cipher.go:165)	LEAVEQ
	0x0184 00388 (/home/frank/arm-128base/src/crypto/des/cipher.go:165)	RET
	0x0185 00389 (/home/frank/arm-128base/src/crypto/des/cipher.go:140)	LEAQ	type:string(SB), AX
	0x018c 00396 (/home/frank/arm-128base/src/crypto/des/cipher.go:140)	LEAQ	crypto/des..stmp_11(SB), BX
	0x0193 00403 (/home/frank/arm-128base/src/crypto/des/cipher.go:140)	PCDATA	$1, $3
	0x0193 00403 (/home/frank/arm-128base/src/crypto/des/cipher.go:140)	CALL	runtime.gopanic(SB)
	0x0198 00408 (/home/frank/arm-128base/src/crypto/des/cipher.go:140)	PCDATA	$3, $1
	0x0198 00408 (/home/frank/arm-128base/src/crypto/des/cipher.go:137)	LEAQ	type:string(SB), AX
	0x019f 00415 (/home/frank/arm-128base/src/crypto/des/cipher.go:137)	LEAQ	crypto/des..stmp_10(SB), BX
	0x01a6 00422 (/home/frank/arm-128base/src/crypto/des/cipher.go:137)	CALL	runtime.gopanic(SB)
	0x01ab 00427 (/home/frank/arm-128base/src/crypto/des/cipher.go:134)	LEAQ	type:string(SB), AX
	0x01b2 00434 (/home/frank/arm-128base/src/crypto/des/cipher.go:134)	LEAQ	crypto/des..stmp_9(SB), BX
	0x01b9 00441 (/home/frank/arm-128base/src/crypto/des/cipher.go:134)	CALL	runtime.gopanic(SB)
	0x01be 00446 (/home/frank/arm-128base/src/crypto/des/cipher.go:134)	XCHGL	AX, AX
	0x01bf 00447 (/home/frank/arm-128base/src/crypto/des/cipher.go:134)	NOP
	0x01bf 00447 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	PCDATA	$1, $-1
	0x01bf 00447 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	PCDATA	$0, $-2
	0x01bf 00447 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	AX, 8(SP)
	0x01c4 00452 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	BX, 16(SP)
	0x01c9 00457 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	CX, 24(SP)
	0x01ce 00462 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	DI, 32(SP)
	0x01d3 00467 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	SI, 40(SP)
	0x01d8 00472 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	R8, 48(SP)
	0x01dd 00477 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	R9, 56(SP)
	0x01e2 00482 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	CALL	runtime.morestack_noctxt(SB)
	0x01e7 00487 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	PCDATA	$0, $-1
	0x01e7 00487 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	8(SP), AX
	0x01ec 00492 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	16(SP), BX
	0x01f1 00497 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	24(SP), CX
	0x01f6 00502 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	32(SP), DI
	0x01fb 00507 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	40(SP), SI
	0x0200 00512 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	48(SP), R8
	0x0205 00517 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	MOVQ	56(SP), R9
	0x020a 00522 (/home/frank/arm-128base/src/crypto/des/cipher.go:132)	JMP	0
	0x0000 49 3b 66 10 0f 86 b5 01 00 00 55 48 89 e5 48 83  I;f.......UH..H.
	0x0010 ec 30 48 89 5c 24 48 48 89 74 24 60 0f 1f 40 00  .0H.\$HH.t$`..@.
	0x0020 49 83 f8 08 0f 8c 81 01 00 00 48 83 f9 08 0f 8c  I.........H.....
	0x0030 64 01 00 00 48 89 44 24 40 48 89 5c 24 48 66 90  d...H.D$@H.\$Hf.
	0x0040 48 39 f3 74 27 48 89 d9 48 8d 56 07 90 48 39 da  H9.t'H..H.V..H9.
	0x0050 72 1a 48 89 f2 48 8d 79 07 0f 1f 80 00 00 00 00  r.H..H.y........
	0x0060 48 39 f7 0f 83 1c 01 00 00 48 89 d6 48 8b 06 48  H9.......H..H..H
	0x0070 0f c8 90 e8 00 00 00 00 48 89 c3 48 c1 eb 20 d1  ........H..H.. .
	0x0080 c3 d1 c0 31 c9 eb 4c 48 89 4c 24 28 48 8b 54 24  ...1..LH.L$(H.T$
	0x0090 40 84 02 48 8d 74 09 f1 48 f7 de 48 8b b4 f2 00  @..H.t..H..H....
	0x00a0 01 00 00 4c 8d 44 09 f2 49 f7 d8 4a 8b bc c2 00  ...L.D..I..J....
	0x00b0 01 00 00 48 89 f1 89 c2 89 d8 89 d3 0f 1f 40 00  ...H..........@.
	0x00c0 e8 00 00 00 00 48 8b 4c 24 28 48 ff c1 89 c2 89  .....H.L$(H.....
	0x00d0 d8 89 d3 48 83 f9 08 7c ae 31 c9 eb 35 48 89 4c  ...H...|.1..5H.L
	0x00e0 24 20 48 8b 54 24 40 84 02 48 89 ce 48 c1 e1 04  $ H.T$@..H..H...
	0x00f0 48 8b 8c 0a 80 00 00 00 48 8d 74 36 01 48 8b bc  H.......H.t6.H..
	0x0100 f2 80 00 00 00 e8 00 00 00 00 48 8b 4c 24 20 48  ..........H.L$ H
	0x0110 ff c1 48 83 f9 08 7c c5 31 c9 eb 44 48 89 4c 24  ..H...|.1..DH.L$
	0x0120 18 48 8b 54 24 40 84 02 48 8d 74 09 f1 48 f7 de  .H.T$@..H.t..H..
	0x0130 48 8b 34 f2 4c 8d 44 09 f2 49 f7 d8 4a 8b 3c c2  H.4.L.D..I..J.<.
	0x0140 48 89 f1 89 c2 89 d8 89 d3 e8 00 00 00 00 48 8b  H.............H.
	0x0150 4c 24 18 48 ff c1 89 c2 89 d8 89 d3 0f 1f 40 00  L$.H..........@.
	0x0160 48 83 f9 08 7c b6 c1 c3 1f c1 c0 1f 48 c1 e0 20  H...|.......H.. 
	0x0170 48 09 d8 e8 00 00 00 00 48 0f c8 48 8b 4c 24 48  H.......H..H.L$H
	0x0180 48 89 01 c9 c3 48 8d 05 00 00 00 00 48 8d 1d 00  H....H......H...
	0x0190 00 00 00 e8 00 00 00 00 48 8d 05 00 00 00 00 48  ........H......H
	0x01a0 8d 1d 00 00 00 00 e8 00 00 00 00 48 8d 05 00 00  ...........H....
	0x01b0 00 00 48 8d 1d 00 00 00 00 e8 00 00 00 00 90 48  ..H............H
	0x01c0 89 44 24 08 48 89 5c 24 10 48 89 4c 24 18 48 89  .D$.H.\$.H.L$.H.
	0x01d0 7c 24 20 48 89 74 24 28 4c 89 44 24 30 4c 89 4c  |$ H.t$(L.D$0L.L
	0x01e0 24 38 e8 00 00 00 00 48 8b 44 24 08 48 8b 5c 24  $8.....H.D$.H.\$
	0x01f0 10 48 8b 4c 24 18 48 8b 7c 24 20 48 8b 74 24 28  .H.L$.H.|$ H.t$(
	0x0200 4c 8b 44 24 30 4c 8b 4c 24 38 e9 f1 fd ff ff     L.D$0L.L$8.....
	rel 3+0 t=R_USEIFACE type:string+0
	rel 3+0 t=R_USEIFACE type:string+0
	rel 3+0 t=R_USEIFACE type:string+0
	rel 116+4 t=R_CALL crypto/des.permuteInitialBlock+0
	rel 193+4 t=R_CALL crypto/des.feistel+0
	rel 262+4 t=R_CALL crypto/des.feistel+0
	rel 330+4 t=R_CALL crypto/des.feistel+0
	rel 372+4 t=R_CALL crypto/des.permuteFinalBlock+0
	rel 392+4 t=R_PCREL type:string+0
	rel 399+4 t=R_PCREL crypto/des..stmp_11+0
	rel 404+4 t=R_CALL runtime.gopanic+0
	rel 411+4 t=R_PCREL type:string+0
	rel 418+4 t=R_PCREL crypto/des..stmp_10+0
	rel 423+4 t=R_CALL runtime.gopanic+0
	rel 430+4 t=R_PCREL type:string+0
	rel 437+4 t=R_PCREL crypto/des..stmp_9+0
	rel 442+4 t=R_CALL runtime.gopanic+0
	rel 483+4 t=R_CALL runtime.morestack_noctxt+0
crypto/des.(*KeySizeError).Error STEXT dupok size=88 align=0x0 args=0x8 locals=0x30 funcid=0x17
	0x0000 00000 (<autogenerated>:1)	TEXT	crypto/des.(*KeySizeError).Error(SB), DUPOK|WRAPPER|ABIInternal, $48-8
	0x0000 00000 (<autogenerated>:1)	CMPQ	SP, 16(R14)
	0x0004 00004 (<autogenerated>:1)	PCDATA	$0, $-2
	0x0004 00004 (<autogenerated>:1)	JLS	71
	0x0006 00006 (<autogenerated>:1)	PCDATA	$0, $-1
	0x0006 00006 (<autogenerated>:1)	PUSHQ	BP
	0x0007 00007 (<autogenerated>:1)	MOVQ	SP, BP
	0x000a 00010 (<autogenerated>:1)	SUBQ	$40, SP
	0x000e 00014 (<autogenerated>:1)	FUNCDATA	$0, gclocals·wvjpxkknJ4nY1JtrArJJaw==(SB)
	0x000e 00014 (<autogenerated>:1)	FUNCDATA	$1, gclocals·J26BEvPExEQhJvjp9E8Whg==(SB)
	0x000e 00014 (<autogenerated>:1)	FUNCDATA	$5, crypto/des.(*KeySizeError).Error.arginfo1(SB)
	0x000e 00014 (<autogenerated>:1)	FUNCDATA	$6, crypto/des.(*KeySizeError).Error.argliveinfo(SB)
	0x000e 00014 (<autogenerated>:1)	PCDATA	$3, $1
	0x000e 00014 (<autogenerated>:1)	TESTQ	AX, AX
	0x0011 00017 (<autogenerated>:1)	JEQ	65
	0x0013 00019 (<autogenerated>:1)	MOVQ	(AX), AX
	0x0016 00022 (<unknown line number>)	NOP
	0x0016 00022 (<unknown line number>)	NOP
	0x0016 00022 (/home/frank/arm-128base/src/strconv/number.go:215)	XCHGL	AX, AX
	0x0017 00023 (/home/frank/arm-128base/src/internal/strconv/itoa.go:53)	MOVL	$10, BX
	0x001c 00028 (/home/frank/arm-128base/src/internal/strconv/itoa.go:53)	PCDATA	$1, $1
	0x001c 00028 (/home/frank/arm-128base/src/internal/strconv/itoa.go:53)	NOP
	0x0020 00032 (/home/frank/arm-128base/src/internal/strconv/itoa.go:53)	CALL	internal/strconv.FormatInt(SB)
	0x0025 00037 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	MOVL	$29, CX
	0x002a 00042 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	MOVQ	AX, DI
	0x002d 00045 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	MOVQ	BX, SI
	0x0030 00048 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	XORL	AX, AX
	0x0032 00050 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	LEAQ	go:string."crypto/des: invalid key size "(SB), BX
	0x0039 00057 (/home/frank/arm-128base/src/crypto/des/cipher.go:22)	CALL	runtime.concatstring2(SB)
	0x003e 00062 (<autogenerated>:1)	LEAVEQ
	0x003f 00063 (<autogenerated>:1)	NOP
	0x0040 00064 (<autogenerated>:1)	RET
	0x0041 00065 (<autogenerated>:1)	CALL	runtime.panicwrap(SB)
	0x0046 00070 (<autogenerated>:1)	XCHGL	AX, AX
	0x0047 00071 (<autogenerated>:1)	NOP
	0x0047 00071 (<autogenerated>:1)	PCDATA	$1, $-1
	0x0047 00071 (<autogenerated>:1)	PCDATA	$0, $-2
	0x0047 00071 (<autogenerated>:1)	MOVQ	AX, 8(SP)
	0x004c 00076 (<autogenerated>:1)	CALL	runtime.morestack_noctxt(SB)
	0x0051 00081 (<autogenerated>:1)	PCDATA	$0, $-1
	0x0051 00081 (<autogenerated>:1)	MOVQ	8(SP), AX
	0x0056 00086 (<autogenerated>:1)	JMP	0
	0x0000 49 3b 66 10 76 41 55 48 89 e5 48 83 ec 28 48 85  I;f.vAUH..H..(H.
	0x0010 c0 74 2e 48 8b 00 90 bb 0a 00 00 00 0f 1f 40 00  .t.H..........@.
	0x0020 e8 00 00 00 00 b9 1d 00 00 00 48 89 c7 48 89 de  ..........H..H..
	0x0030 31 c0 48 8d 1d 00 00 00 00 e8 00 00 00 00 c9 90  1.H.............
	0x0040 c3 e8 00 00 00 00 90 48 89 44 24 08 e8 00 00 00  .......H.D$.....
	0x0050 00 48 8b 44 24 08 eb a8                          .H.D$...
	rel 33+4 t=R_CALL internal/strconv.FormatInt+0
	rel 53+4 t=R_PCREL go:string."crypto/des: invalid key size "+0
	rel 58+4 t=R_CALL runtime.concatstring2+0
	rel 66+4 t=R_CALL runtime.panicwrap+0
	rel 77+4 t=R_CALL runtime.morestack_noctxt+0
go:cuinfo.producer.crypto/des SDWARFCUINFO dupok size=0 align=0x0
	0x0000 72 65 67 61 62 69                                regabi
runtime.memequal64·f SRODATA dupok size=8 align=0x0
	0x0000 00 00 00 00 00 00 00 00                          ........
	rel 0+8 t=R_ADDR runtime.memequal64+0
runtime.gcbits.0100000000000000 SRODATA dupok size=8 align=0x8
	0x0000 01 00 00 00 00 00 00 00                          ........
type:.namedata.*des.KeySizeError. SRODATA dupok size=19 align=0x1
	0x0000 01 11 2a 64 65 73 2e 4b 65 79 53 69 7a 65 45 72  ..*des.KeySizeEr
	0x0010 72 6f 72                                         ror
type:.namedata.*func(*des.KeySizeError) string- SRODATA dupok size=33 align=0x1
	0x0000 00 1f 2a 66 75 6e 63 28 2a 64 65 73 2e 4b 65 79  ..*func(*des.Key
	0x0010 53 69 7a 65 45 72 72 6f 72 29 20 73 74 72 69 6e  SizeError) strin
	0x0020 67                                               g
type:*func(*crypto/des.KeySizeError) string SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 c7 3e 82 85 28 08 08 16 00 00 00 00 00 00 00 00  .>..(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.KeySizeError) string-+0
	rel 48+8 t=R_ADDR type:func(*crypto/des.KeySizeError) string+0
type:func(*crypto/des.KeySizeError) string SRODATA dupok size=72 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 f8 c0 d9 42 22 08 08 13 00 00 00 00 00 00 00 00  ...B"...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 01 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.KeySizeError) string-+0
	rel 44+4 t=RelocType(-32763) type:*func(*crypto/des.KeySizeError) string+0
	rel 56+8 t=R_ADDR type:*crypto/des.KeySizeError+0
	rel 64+8 t=R_ADDR type:string+0
type:.importpath.crypto/des. SRODATA dupok size=12 align=0x1
	0x0000 00 0a 63 72 79 70 74 6f 2f 64 65 73              ..crypto/des
type:.namedata.Error. SRODATA dupok size=7 align=0x1
	0x0000 01 05 45 72 72 6f 72                             ..Error
type:.namedata.*func() string- SRODATA dupok size=16 align=0x1
	0x0000 00 0e 2a 66 75 6e 63 28 29 20 73 74 72 69 6e 67  ..*func() string
type:*func() string SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 38 65 5b 28 28 08 08 16 00 00 00 00 00 00 00 00  8e[((...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func() string-+0
	rel 48+8 t=R_ADDR type:func() string+0
type:func() string SRODATA dupok size=64 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 01 1f 55 63 22 08 08 13 00 00 00 00 00 00 00 00  ..Uc"...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func() string-+0
	rel 44+4 t=RelocType(-32763) type:*func() string+0
	rel 56+8 t=R_ADDR type:string+0
type:*crypto/des.KeySizeError SRODATA size=88 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 99 22 39 8f 29 08 08 16 00 00 00 00 00 00 00 00  ."9.)...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 01 00 01 00  ................
	0x0040 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0050 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*des.KeySizeError.+0
	rel 48+8 t=R_ADDR type:crypto/des.KeySizeError+0
	rel 56+4 t=R_ADDROFF type:.importpath.crypto/des.+0
	rel 72+4 t=R_ADDROFF type:.namedata.Error.+0
	rel 76+4 t=R_METHODOFF type:func() string+0
	rel 80+4 t=R_METHODOFF crypto/des.(*KeySizeError).Error+0
	rel 84+4 t=R_METHODOFF crypto/des.(*KeySizeError).Error+0
runtime.gcbits. SRODATA dupok size=0 align=0x8
type:.namedata.*func(des.KeySizeError) string- SRODATA dupok size=32 align=0x1
	0x0000 00 1e 2a 66 75 6e 63 28 64 65 73 2e 4b 65 79 53  ..*func(des.KeyS
	0x0010 69 7a 65 45 72 72 6f 72 29 20 73 74 72 69 6e 67  izeError) string
type:*func(crypto/des.KeySizeError) string SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 93 35 1c 52 28 08 08 16 00 00 00 00 00 00 00 00  .5.R(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(des.KeySizeError) string-+0
	rel 48+8 t=R_ADDR type:func(crypto/des.KeySizeError) string+0
type:func(crypto/des.KeySizeError) string SRODATA dupok size=72 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 1f 37 e5 07 22 08 08 13 00 00 00 00 00 00 00 00  .7.."...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 01 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(des.KeySizeError) string-+0
	rel 44+4 t=RelocType(-32763) type:*func(crypto/des.KeySizeError) string+0
	rel 56+8 t=R_ADDR type:crypto/des.KeySizeError+0
	rel 64+8 t=R_ADDR type:string+0
type:crypto/des.KeySizeError SRODATA size=80 align=0x8
	0x0000 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 7d 1e b8 4d 0f 08 08 02 00 00 00 00 00 00 00 00  }..M............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 01 00 01 00 10 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*des.KeySizeError.+0
	rel 44+4 t=R_ADDROFF type:*crypto/des.KeySizeError+0
	rel 48+4 t=R_ADDROFF type:.importpath.crypto/des.+0
	rel 64+4 t=R_ADDROFF type:.namedata.Error.+0
	rel 68+4 t=R_METHODOFF type:func() string+0
	rel 72+4 t=R_METHODOFF crypto/des.(*KeySizeError).Error+0
	rel 76+4 t=R_METHODOFF crypto/des.KeySizeError.Error+0
go:itab.crypto/des.KeySizeError,error SRODATA dupok size=32 align=0x8
	0x0000 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 7d 1e b8 4d 00 00 00 00 00 00 00 00 00 00 00 00  }..M............
	rel 0+8 t=R_ADDR type:error+0
	rel 8+8 t=R_ADDR type:crypto/des.KeySizeError+0
	rel 24+8 t=RelocType(-32767) crypto/des.(*KeySizeError).Error+0
type:.namedata.*des.tripleDESCipher- SRODATA dupok size=22 align=0x1
	0x0000 00 14 2a 64 65 73 2e 74 72 69 70 6c 65 44 45 53  ..*des.tripleDES
	0x0010 43 69 70 68 65 72                                Cipher
type:.eqfunc.M384 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 80 01 00 00 00 00 00 00  ................
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.eqfunc.M128 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 80 00 00 00 00 00 00 00  ................
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.namedata.*des.desCipher- SRODATA dupok size=16 align=0x1
	0x0000 00 0e 2a 64 65 73 2e 64 65 73 43 69 70 68 65 72  ..*des.desCipher
type:.namedata.*func(*des.desCipher) int- SRODATA dupok size=27 align=0x1
	0x0000 00 19 2a 66 75 6e 63 28 2a 64 65 73 2e 64 65 73  ..*func(*des.des
	0x0010 43 69 70 68 65 72 29 20 69 6e 74                 Cipher) int
type:*func(*crypto/des.desCipher) int SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 79 09 82 0c 28 08 08 16 00 00 00 00 00 00 00 00  y...(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.desCipher) int-+0
	rel 48+8 t=R_ADDR type:func(*crypto/des.desCipher) int+0
type:func(*crypto/des.desCipher) int SRODATA dupok size=72 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 0f 61 0a 3b 22 08 08 13 00 00 00 00 00 00 00 00  .a.;"...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 01 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.desCipher) int-+0
	rel 44+4 t=RelocType(-32763) type:*func(*crypto/des.desCipher) int+0
	rel 56+8 t=R_ADDR type:*crypto/des.desCipher+0
	rel 64+8 t=R_ADDR type:int+0
type:.namedata.*func(*des.desCipher, []uint8, []uint8)- SRODATA dupok size=41 align=0x1
	0x0000 00 27 2a 66 75 6e 63 28 2a 64 65 73 2e 64 65 73  .'*func(*des.des
	0x0010 43 69 70 68 65 72 2c 20 5b 5d 75 69 6e 74 38 2c  Cipher, []uint8,
	0x0020 20 5b 5d 75 69 6e 74 38 29                        []uint8)
type:*func(*crypto/des.desCipher, []uint8, []uint8) SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 66 73 13 d0 28 08 08 16 00 00 00 00 00 00 00 00  fs..(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.desCipher, []uint8, []uint8)-+0
	rel 48+8 t=R_ADDR type:func(*crypto/des.desCipher, []uint8, []uint8)+0
type:func(*crypto/des.desCipher, []uint8, []uint8) SRODATA dupok size=80 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 27 ca 10 34 22 08 08 13 00 00 00 00 00 00 00 00  '..4"...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.desCipher, []uint8, []uint8)-+0
	rel 44+4 t=RelocType(-32763) type:*func(*crypto/des.desCipher, []uint8, []uint8)+0
	rel 56+8 t=R_ADDR type:*crypto/des.desCipher+0
	rel 64+8 t=R_ADDR type:[]uint8+0
	rel 72+8 t=R_ADDR type:[]uint8+0
type:.namedata.*func(*des.desCipher, []uint8)- SRODATA dupok size=32 align=0x1
	0x0000 00 1e 2a 66 75 6e 63 28 2a 64 65 73 2e 64 65 73  ..*func(*des.des
	0x0010 43 69 70 68 65 72 2c 20 5b 5d 75 69 6e 74 38 29  Cipher, []uint8)
type:*func(*crypto/des.desCipher, []uint8) SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 26 c7 e1 7c 28 08 08 16 00 00 00 00 00 00 00 00  &..|(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.desCipher, []uint8)-+0
	rel 48+8 t=R_ADDR type:func(*crypto/des.desCipher, []uint8)+0
type:func(*crypto/des.desCipher, []uint8) SRODATA dupok size=72 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 a3 a8 4a bb 22 08 08 13 00 00 00 00 00 00 00 00  ..J."...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.desCipher, []uint8)-+0
	rel 44+4 t=RelocType(-32763) type:*func(*crypto/des.desCipher, []uint8)+0
	rel 56+8 t=R_ADDR type:*crypto/des.desCipher+0
	rel 64+8 t=R_ADDR type:[]uint8+0
type:.namedata.BlockSize. SRODATA dupok size=11 align=0x1
	0x0000 01 09 42 6c 6f 63 6b 53 69 7a 65                 ..BlockSize
type:.namedata.*func() int- SRODATA dupok size=13 align=0x1
	0x0000 00 0b 2a 66 75 6e 63 28 29 20 69 6e 74           ..*func() int
type:*func() int SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 a8 d2 35 e7 28 08 08 16 00 00 00 00 00 00 00 00  ..5.(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func() int-+0
	rel 48+8 t=R_ADDR type:func() int+0
type:func() int SRODATA dupok size=64 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 bd 97 29 21 22 08 08 13 00 00 00 00 00 00 00 00  ..)!"...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func() int-+0
	rel 44+4 t=RelocType(-32763) type:*func() int+0
	rel 56+8 t=R_ADDR type:int+0
type:.namedata.Decrypt. SRODATA dupok size=9 align=0x1
	0x0000 01 07 44 65 63 72 79 70 74                       ..Decrypt
type:.namedata.*func([]uint8, []uint8)- SRODATA dupok size=25 align=0x1
	0x0000 00 17 2a 66 75 6e 63 28 5b 5d 75 69 6e 74 38 2c  ..*func([]uint8,
	0x0010 20 5b 5d 75 69 6e 74 38 29                        []uint8)
type:*func([]uint8, []uint8) SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 9b 52 41 0b 28 08 08 16 00 00 00 00 00 00 00 00  .RA.(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func([]uint8, []uint8)-+0
	rel 48+8 t=R_ADDR type:func([]uint8, []uint8)+0
type:func([]uint8, []uint8) SRODATA dupok size=72 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 81 33 7a c4 22 08 08 13 00 00 00 00 00 00 00 00  .3z."...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func([]uint8, []uint8)-+0
	rel 44+4 t=RelocType(-32763) type:*func([]uint8, []uint8)+0
	rel 56+8 t=R_ADDR type:[]uint8+0
	rel 64+8 t=R_ADDR type:[]uint8+0
type:.namedata.Encrypt. SRODATA dupok size=9 align=0x1
	0x0000 01 07 45 6e 63 72 79 70 74                       ..Encrypt
type:.namedata.generateSubkeys- SRODATA dupok size=17 align=0x1
	0x0000 00 0f 67 65 6e 65 72 61 74 65 53 75 62 6b 65 79  ..generateSubkey
	0x0010 73                                               s
type:.namedata.*func([]uint8)- SRODATA dupok size=16 align=0x1
	0x0000 00 0e 2a 66 75 6e 63 28 5b 5d 75 69 6e 74 38 29  ..*func([]uint8)
type:*func([]uint8) SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 ac f4 b8 e9 28 08 08 16 00 00 00 00 00 00 00 00  ....(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func([]uint8)-+0
	rel 48+8 t=R_ADDR type:func([]uint8)+0
type:func([]uint8) SRODATA dupok size=64 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 19 57 76 b0 22 08 08 13 00 00 00 00 00 00 00 00  .Wv."...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func([]uint8)-+0
	rel 44+4 t=RelocType(-32763) type:*func([]uint8)+0
	rel 56+8 t=R_ADDR type:[]uint8+0
type:*crypto/des.desCipher SRODATA size=136 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 b0 93 13 a6 29 08 08 16 00 00 00 00 00 00 00 00  ....)...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 04 00 03 00  ................
	0x0040 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0050 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0080 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*des.desCipher-+0
	rel 48+8 t=R_ADDR type:crypto/des.desCipher+0
	rel 56+4 t=R_ADDROFF type:.importpath.crypto/des.+0
	rel 72+4 t=R_ADDROFF type:.namedata.BlockSize.+0
	rel 76+4 t=R_METHODOFF type:func() int+0
	rel 80+4 t=R_METHODOFF crypto/des.(*desCipher).BlockSize+0
	rel 84+4 t=R_METHODOFF crypto/des.(*desCipher).BlockSize+0
	rel 88+4 t=R_ADDROFF type:.namedata.Decrypt.+0
	rel 92+4 t=R_METHODOFF type:func([]uint8, []uint8)+0
	rel 96+4 t=R_METHODOFF crypto/des.(*desCipher).Decrypt+0
	rel 100+4 t=R_METHODOFF crypto/des.(*desCipher).Decrypt+0
	rel 104+4 t=R_ADDROFF type:.namedata.Encrypt.+0
	rel 108+4 t=R_METHODOFF type:func([]uint8, []uint8)+0
	rel 112+4 t=R_METHODOFF crypto/des.(*desCipher).Encrypt+0
	rel 116+4 t=R_METHODOFF crypto/des.(*desCipher).Encrypt+0
	rel 120+4 t=R_ADDROFF type:.namedata.generateSubkeys-+0
	rel 124+4 t=R_METHODOFF type:func([]uint8)+0
	rel 128+4 t=R_METHODOFF crypto/des.(*desCipher).generateSubkeys+0
	rel 132+4 t=R_METHODOFF crypto/des.(*desCipher).generateSubkeys+0
type:.namedata.*[16]uint64- SRODATA dupok size=13 align=0x1
	0x0000 00 0b 2a 5b 31 36 5d 75 69 6e 74 36 34           ..*[16]uint64
type:*[16]uint64 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 8a 08 ee fe 28 08 08 16 00 00 00 00 00 00 00 00  ....(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[16]uint64-+0
	rel 48+8 t=R_ADDR type:[16]uint64+0
type:[16]uint64 SRODATA dupok size=72 align=0x8
	0x0000 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 7c ab fb 0d 0a 08 08 11 00 00 00 00 00 00 00 00  |...............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 10 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR type:.eqfunc.M128+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[16]uint64-+0
	rel 44+4 t=RelocType(-32763) type:*[16]uint64+0
	rel 48+8 t=R_ADDR type:uint64+0
	rel 56+8 t=R_ADDR type:[]uint64+0
type:.namedata.subkeys- SRODATA dupok size=9 align=0x1
	0x0000 00 07 73 75 62 6b 65 79 73                       ..subkeys
type:crypto/des.desCipher SRODATA size=120 align=0x8
	0x0000 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 2d 71 95 b9 0f 08 08 19 00 00 00 00 00 00 00 00  -q..............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00  ................
	0x0050 00 00 00 00 00 00 00 00 28 00 00 00 00 00 00 00  ........(.......
	0x0060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0070 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR type:.eqfunc.M128+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*des.desCipher-+0
	rel 44+4 t=R_ADDROFF type:*crypto/des.desCipher+0
	rel 48+8 t=R_ADDR type:.importpath.crypto/des.+0
	rel 56+8 t=R_ADDR type:crypto/des.desCipher+96
	rel 80+4 t=R_ADDROFF type:.importpath.crypto/des.+0
	rel 96+8 t=R_ADDR type:.namedata.subkeys-+0
	rel 104+8 t=R_ADDR type:[16]uint64+0
type:.namedata.cipher1- SRODATA dupok size=9 align=0x1
	0x0000 00 07 63 69 70 68 65 72 31                       ..cipher1
type:.namedata.cipher2- SRODATA dupok size=9 align=0x1
	0x0000 00 07 63 69 70 68 65 72 32                       ..cipher2
type:.namedata.cipher3- SRODATA dupok size=9 align=0x1
	0x0000 00 07 63 69 70 68 65 72 33                       ..cipher3
type:crypto/des.tripleDESCipher SRODATA size=168 align=0x8
	0x0000 80 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 c8 db 72 f7 0f 08 08 19 00 00 00 00 00 00 00 00  ..r.............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 03 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00  ................
	0x0050 00 00 00 00 00 00 00 00 58 00 00 00 00 00 00 00  ........X.......
	0x0060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0080 00 00 00 00 00 00 00 00 80 00 00 00 00 00 00 00  ................
	0x0090 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x00a0 00 01 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR type:.eqfunc.M384+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*des.tripleDESCipher-+0
	rel 44+4 t=R_ADDROFF type:*crypto/des.tripleDESCipher+0
	rel 48+8 t=R_ADDR type:.importpath.crypto/des.+0
	rel 56+8 t=R_ADDR type:crypto/des.tripleDESCipher+96
	rel 80+4 t=R_ADDROFF type:.importpath.crypto/des.+0
	rel 96+8 t=R_ADDR type:.namedata.cipher1-+0
	rel 104+8 t=R_ADDR type:crypto/des.desCipher+0
	rel 120+8 t=R_ADDR type:.namedata.cipher2-+0
	rel 128+8 t=R_ADDR type:crypto/des.desCipher+0
	rel 144+8 t=R_ADDR type:.namedata.cipher3-+0
	rel 152+8 t=R_ADDR type:crypto/des.desCipher+0
type:.namedata.*func(*des.tripleDESCipher) int- SRODATA dupok size=33 align=0x1
	0x0000 00 1f 2a 66 75 6e 63 28 2a 64 65 73 2e 74 72 69  ..*func(*des.tri
	0x0010 70 6c 65 44 45 53 43 69 70 68 65 72 29 20 69 6e  pleDESCipher) in
	0x0020 74                                               t
type:*func(*crypto/des.tripleDESCipher) int SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 86 84 79 e5 28 08 08 16 00 00 00 00 00 00 00 00  ..y.(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.tripleDESCipher) int-+0
	rel 48+8 t=R_ADDR type:func(*crypto/des.tripleDESCipher) int+0
type:func(*crypto/des.tripleDESCipher) int SRODATA dupok size=72 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 ef d2 55 60 22 08 08 13 00 00 00 00 00 00 00 00  ..U`"...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 01 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.tripleDESCipher) int-+0
	rel 44+4 t=RelocType(-32763) type:*func(*crypto/des.tripleDESCipher) int+0
	rel 56+8 t=R_ADDR type:*crypto/des.tripleDESCipher+0
	rel 64+8 t=R_ADDR type:int+0
type:.namedata.*func(*des.tripleDESCipher, []uint8, []uint8)- SRODATA dupok size=47 align=0x1
	0x0000 00 2d 2a 66 75 6e 63 28 2a 64 65 73 2e 74 72 69  .-*func(*des.tri
	0x0010 70 6c 65 44 45 53 43 69 70 68 65 72 2c 20 5b 5d  pleDESCipher, []
	0x0020 75 69 6e 74 38 2c 20 5b 5d 75 69 6e 74 38 29     uint8, []uint8)
type:*func(*crypto/des.tripleDESCipher, []uint8, []uint8) SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 06 a8 88 7e 28 08 08 16 00 00 00 00 00 00 00 00  ...~(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.tripleDESCipher, []uint8, []uint8)-+0
	rel 48+8 t=R_ADDR type:func(*crypto/des.tripleDESCipher, []uint8, []uint8)+0
type:func(*crypto/des.tripleDESCipher, []uint8, []uint8) SRODATA dupok size=80 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 00 7c 16 2e 22 08 08 13 00 00 00 00 00 00 00 00  .|.."...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func(*des.tripleDESCipher, []uint8, []uint8)-+0
	rel 44+4 t=RelocType(-32763) type:*func(*crypto/des.tripleDESCipher, []uint8, []uint8)+0
	rel 56+8 t=R_ADDR type:*crypto/des.tripleDESCipher+0
	rel 64+8 t=R_ADDR type:[]uint8+0
	rel 72+8 t=R_ADDR type:[]uint8+0
type:*crypto/des.tripleDESCipher SRODATA size=120 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 b0 d5 7c 98 29 08 08 16 00 00 00 00 00 00 00 00  ..|.)...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 03 00 03 00  ................
	0x0040 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0050 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0070 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*des.tripleDESCipher-+0
	rel 48+8 t=R_ADDR type:crypto/des.tripleDESCipher+0
	rel 56+4 t=R_ADDROFF type:.importpath.crypto/des.+0
	rel 72+4 t=R_ADDROFF type:.namedata.BlockSize.+0
	rel 76+4 t=R_METHODOFF type:func() int+0
	rel 80+4 t=R_METHODOFF crypto/des.(*tripleDESCipher).BlockSize+0
	rel 84+4 t=R_METHODOFF crypto/des.(*tripleDESCipher).BlockSize+0
	rel 88+4 t=R_ADDROFF type:.namedata.Decrypt.+0
	rel 92+4 t=R_METHODOFF type:func([]uint8, []uint8)+0
	rel 96+4 t=R_METHODOFF crypto/des.(*tripleDESCipher).Decrypt+0
	rel 100+4 t=R_METHODOFF crypto/des.(*tripleDESCipher).Decrypt+0
	rel 104+4 t=R_ADDROFF type:.namedata.Encrypt.+0
	rel 108+4 t=R_METHODOFF type:func([]uint8, []uint8)+0
	rel 112+4 t=R_METHODOFF crypto/des.(*tripleDESCipher).Encrypt+0
	rel 116+4 t=R_METHODOFF crypto/des.(*tripleDESCipher).Encrypt+0
go:itab.*crypto/des.tripleDESCipher,crypto/cipher.Block SRODATA dupok size=48 align=0x8
	0x0000 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 b0 d5 7c 98 00 00 00 00 00 00 00 00 00 00 00 00  ..|.............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 0+8 t=R_ADDR type:crypto/cipher.Block+0
	rel 8+8 t=R_ADDR type:*crypto/des.tripleDESCipher+0
	rel 24+8 t=RelocType(-32767) crypto/des.(*tripleDESCipher).BlockSize+0
	rel 32+8 t=RelocType(-32767) crypto/des.(*tripleDESCipher).Decrypt+0
	rel 40+8 t=RelocType(-32767) crypto/des.(*tripleDESCipher).Encrypt+0
go:itab.*crypto/des.desCipher,crypto/cipher.Block SRODATA dupok size=48 align=0x8
	0x0000 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 b0 93 13 a6 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 0+8 t=R_ADDR type:crypto/cipher.Block+0
	rel 8+8 t=R_ADDR type:*crypto/des.desCipher+0
	rel 24+8 t=RelocType(-32767) crypto/des.(*desCipher).BlockSize+0
	rel 32+8 t=RelocType(-32767) crypto/des.(*desCipher).Decrypt+0
	rel 40+8 t=RelocType(-32767) crypto/des.(*desCipher).Encrypt+0
go:cuinfo.packagename.crypto/des SDWARFCUINFO dupok size=0 align=0x0
	0x0000 64 65 73                                         des
go:info.internal/byteorder.BEUint64$abstract SDWARFABSFCN dupok size=41 align=0x0
	0x0000 05 69 6e 74 65 72 6e 61 6c 2f 62 79 74 65 6f 72  .internal/byteor
	0x0010 64 65 72 2e 42 45 55 69 6e 74 36 34 00 01 78 01  der.BEUint64..x.
	0x0020 22 62 00 00 00 00 00 00 00                       "b.......
	rel 0+0 t=R_USETYPE type:[]uint8+0
	rel 0+0 t=R_USETYPE type:uint64+0
	rel 36+4 t=R_DWARFSECREF go:info.[]uint8+0
go:info.internal/byteorder.BEPutUint64$abstract SDWARFABSFCN dupok size=52 align=0x0
	0x0000 05 69 6e 74 65 72 6e 61 6c 2f 62 79 74 65 6f 72  .internal/byteor
	0x0010 64 65 72 2e 42 45 50 75 74 55 69 6e 74 36 34 00  der.BEPutUint64.
	0x0020 01 7e 01 22 62 00 00 00 00 00 00 22 76 00 00 00  .~."b......"v...
	0x0030 00 00 00 00                                      ....
	rel 0+0 t=R_USETYPE type:[]uint8+0
	rel 0+0 t=R_USETYPE type:uint64+0
	rel 39+4 t=R_DWARFSECREF go:info.[]uint8+0
	rel 47+4 t=R_DWARFSECREF go:info.uint64+0
go:info.crypto/des.permuteBlock$abstract SDWARFABSFCN dupok size=102 align=0x0
	0x0000 05 63 72 79 70 74 6f 2f 64 65 73 2e 70 65 72 6d  .crypto/des.perm
	0x0010 75 74 65 42 6c 6f 63 6b 00 01 4d 01 22 73 72 63  uteBlock..M."src
	0x0020 00 00 00 00 00 00 22 70 65 72 6d 75 74 61 74 69  ......"permutati
	0x0030 6f 6e 00 00 00 00 00 00 22 62 6c 6f 63 6b 00 01  on......"block..
	0x0040 00 00 00 00 21 70 6f 73 69 74 69 6f 6e 00 4e 00  ....!position.N.
	0x0050 00 00 00 21 6e 00 4e 00 00 00 00 21 62 69 74 00  ...!n.N....!bit.
	0x0060 4f 00 00 00 00 00                                O.....
	rel 34+4 t=R_DWARFSECREF go:info.uint64+0
	rel 52+4 t=R_DWARFSECREF go:info.[]uint8+0
	rel 64+4 t=R_DWARFSECREF go:info.uint64+0
	rel 79+4 t=R_DWARFSECREF go:info.int+0
	rel 87+4 t=R_DWARFSECREF go:info.uint8+0
	rel 97+4 t=R_DWARFSECREF go:info.uint64+0
go:info.sync.(*Once).Do$abstract SDWARFABSFCN dupok size=37 align=0x0
	0x0000 05 73 79 6e 63 2e 28 2a 4f 6e 63 65 29 2e 44 6f  .sync.(*Once).Do
	0x0010 00 01 34 01 22 6f 00 00 00 00 00 00 22 66 00 00  ..4."o......"f..
	0x0020 00 00 00 00 00                                   .....
	rel 0+0 t=R_USETYPE type:*sync.Once+0
	rel 0+0 t=R_USETYPE type:func()+0
	rel 24+4 t=R_DWARFSECREF go:info.*sync.Once+0
	rel 32+4 t=R_DWARFSECREF go:info.func()+0
go:info.crypto/des.ksRotate$abstract SDWARFABSFCN dupok size=91 align=0x0
	0x0000 05 63 72 79 70 74 6f 2f 64 65 73 2e 6b 73 52 6f  .crypto/des.ksRo
	0x0010 74 61 74 65 00 01 cb 01 01 22 69 6e 00 00 00 00  tate....."in....
	0x0020 00 00 22 6f 75 74 00 01 00 00 00 00 21 6c 61 73  .."out......!las
	0x0030 74 00 cd 01 00 00 00 00 21 69 00 ce 01 00 00 00  t.......!i......
	0x0040 00 21 6c 65 66 74 00 d0 01 00 00 00 00 21 72 69  .!left.......!ri
	0x0050 67 68 74 00 d1 01 00 00 00 00 00                 ght........
	rel 30+4 t=R_DWARFSECREF go:info.uint32+0
	rel 40+4 t=R_DWARFSECREF go:info.[]uint32+0
	rel 52+4 t=R_DWARFSECREF go:info.uint32+0
	rel 61+4 t=R_DWARFSECREF go:info.int+0
	rel 73+4 t=R_DWARFSECREF go:info.uint32+0
	rel 86+4 t=R_DWARFSECREF go:info.uint32+0
go:info.crypto/des.unpack$abstract SDWARFABSFCN dupok size=32 align=0x0
	0x0000 05 63 72 79 70 74 6f 2f 64 65 73 2e 75 6e 70 61  .crypto/des.unpa
	0x0010 63 6b 00 01 f0 01 01 22 78 00 00 00 00 00 00 00  ck....."x.......
	rel 27+4 t=R_DWARFSECREF go:info.uint64+0
go:info.sync/atomic.(*Bool).Load$abstract SDWARFABSFCN dupok size=38 align=0x0
	0x0000 05 73 79 6e 63 2f 61 74 6f 6d 69 63 2e 28 2a 42  .sync/atomic.(*B
	0x0010 6f 6f 6c 29 2e 4c 6f 61 64 00 01 13 01 22 78 00  ool).Load...."x.
	0x0020 00 00 00 00 00 00                                ......
	rel 0+0 t=R_USETYPE type:*sync/atomic.Bool+0
	rel 0+0 t=R_USETYPE type:bool+0
	rel 33+4 t=R_DWARFSECREF go:info.*sync/atomic.Bool+0
go:info.strconv.Itoa$abstract SDWARFABSFCN dupok size=27 align=0x0
	0x0000 05 73 74 72 63 6f 6e 76 2e 49 74 6f 61 00 01 d6  .strconv.Itoa...
	0x0010 01 01 22 69 00 00 00 00 00 00 00                 .."i.......
	rel 0+0 t=R_USETYPE type:int+0
	rel 0+0 t=R_USETYPE type:string+0
	rel 22+4 t=R_DWARFSECREF go:info.int+0
go:info.internal/strconv.Itoa$abstract SDWARFABSFCN dupok size=35 align=0x0
	0x0000 05 69 6e 74 65 72 6e 61 6c 2f 73 74 72 63 6f 6e  .internal/strcon
	0x0010 76 2e 49 74 6f 61 00 01 34 01 22 69 00 00 00 00  v.Itoa..4."i....
	0x0020 00 00 00                                         ...
	rel 0+0 t=R_USETYPE type:int+0
	rel 0+0 t=R_USETYPE type:string+0
	rel 30+4 t=R_DWARFSECREF go:info.int+0
go:info.crypto/internal/fips140only.Enforced$abstract SDWARFABSFCN dupok size=42 align=0x0
	0x0000 05 63 72 79 70 74 6f 2f 69 6e 74 65 72 6e 61 6c  .crypto/internal
	0x0010 2f 66 69 70 73 31 34 30 6f 6e 6c 79 2e 45 6e 66  /fips140only.Enf
	0x0020 6f 72 63 65 64 00 01 11 01 00                    orced.....
	rel 0+0 t=R_USETYPE type:bool+0
go:info.errors.New$abstract SDWARFABSFCN dupok size=27 align=0x0
	0x0000 05 65 72 72 6f 72 73 2e 4e 65 77 00 01 40 01 22  .errors.New..@."
	0x0010 74 65 78 74 00 00 00 00 00 00 00                 text.......
	rel 0+0 t=R_USETYPE type:error+0
	rel 0+0 t=R_USETYPE type:string+0
	rel 22+4 t=R_DWARFSECREF go:info.string+0
go:itab.*errors.errorString,error SRODATA dupok size=32 align=0x8
	0x0000 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 a0 d2 33 8f 00 00 00 00 00 00 00 00 00 00 00 00  ..3.............
	rel 0+8 t=R_ADDR type:error+0
	rel 8+8 t=R_ADDR type:*errors.errorString+0
	rel 24+8 t=RelocType(-32767) errors.(*errorString).Error+0
go:info.crypto/fips140.Enforced$abstract SDWARFABSFCN dupok size=29 align=0x0
	0x0000 05 63 72 79 70 74 6f 2f 66 69 70 73 31 34 30 2e  .crypto/fips140.
	0x0010 45 6e 66 6f 72 63 65 64 00 01 24 01 00           Enforced..$..
	rel 0+0 t=R_USETYPE type:bool+0
go:info.crypto/internal/fips140/alias.InexactOverlap$abstract SDWARFABSFCN dupok size=66 align=0x0
	0x0000 05 63 72 79 70 74 6f 2f 69 6e 74 65 72 6e 61 6c  .crypto/internal
	0x0010 2f 66 69 70 73 31 34 30 2f 61 6c 69 61 73 2e 49  /fips140/alias.I
	0x0020 6e 65 78 61 63 74 4f 76 65 72 6c 61 70 00 01 19  nexactOverlap...
	0x0030 01 22 78 00 00 00 00 00 00 22 79 00 00 00 00 00  ."x......"y.....
	0x0040 00 00                                            ..
	rel 0+0 t=R_USETYPE type:[]uint8+0
	rel 0+0 t=R_USETYPE type:bool+0
	rel 53+4 t=R_DWARFSECREF go:info.[]uint8+0
	rel 61+4 t=R_DWARFSECREF go:info.[]uint8+0
go:info.crypto/internal/fips140/alias.AnyOverlap$abstract SDWARFABSFCN dupok size=62 align=0x0
	0x0000 05 63 72 79 70 74 6f 2f 69 6e 74 65 72 6e 61 6c  .crypto/internal
	0x0010 2f 66 69 70 73 31 34 30 2f 61 6c 69 61 73 2e 41  /fips140/alias.A
	0x0020 6e 79 4f 76 65 72 6c 61 70 00 01 0d 01 22 78 00  nyOverlap...."x.
	0x0030 00 00 00 00 00 22 79 00 00 00 00 00 00 00        ....."y.......
	rel 0+0 t=R_USETYPE type:[]uint8+0
	rel 0+0 t=R_USETYPE type:bool+0
	rel 49+4 t=R_DWARFSECREF go:info.[]uint8+0
	rel 57+4 t=R_DWARFSECREF go:info.[]uint8+0
go:info.crypto/des.KeySizeError.Error$abstract SDWARFABSFCN dupok size=43 align=0x0
	0x0000 05 63 72 79 70 74 6f 2f 64 65 73 2e 4b 65 79 53  .crypto/des.KeyS
	0x0010 69 7a 65 45 72 72 6f 72 2e 45 72 72 6f 72 00 01  izeError.Error..
	0x0020 15 01 22 6b 00 00 00 00 00 00 00                 .."k.......
	rel 38+4 t=R_DWARFSECREF go:info.crypto/des.KeySizeError+0
crypto/des..inittask SNOPTRDATA size=8 align=0x0
	0x0000 00 00 00 00 00 00 00 00                          ........
	rel 0+0 t=R_INITORDER sync..inittask+0
	rel 0+0 t=R_INITORDER crypto/cipher..inittask+0
	rel 0+0 t=R_INITORDER crypto/internal/fips140only..inittask+0
	rel 0+0 t=R_INITORDER errors..inittask+0
	rel 0+0 t=R_INITORDER strconv..inittask+0
crypto/des.feistelBox SNOPTRBSS size=2048 align=0x0
 SDWARFVAR size=38 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 66 65 69 73  .crypto/des.feis
	0x0010 74 65 6c 42 6f 78 00 09 03 00 00 00 00 00 00 00  telBox..........
	0x0020 00 00 00 00 00 01                                ......
	rel 25+8 t=R_ADDR crypto/des.feistelBox+0
	rel 33+4 t=R_DWARFSECREF go:info.[8][64]uint32+0
type:.eqfunc.M2048 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 00 08 00 00 00 00 00 00  ................
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.namedata.*[8][64]uint32- SRODATA dupok size=16 align=0x1
	0x0000 00 0e 2a 5b 38 5d 5b 36 34 5d 75 69 6e 74 33 32  ..*[8][64]uint32
type:*[8][64]uint32 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 9c 65 3a 05 28 08 08 16 00 00 00 00 00 00 00 00  .e:.(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[8][64]uint32-+0
	rel 48+8 t=R_ADDR type:[8][64]uint32+0
type:.eqfunc.M256 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00 00  ................
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.namedata.*[64]uint32- SRODATA dupok size=13 align=0x1
	0x0000 00 0b 2a 5b 36 34 5d 75 69 6e 74 33 32           ..*[64]uint32
type:*[64]uint32 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 68 a7 69 17 28 08 08 16 00 00 00 00 00 00 00 00  h.i.(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[64]uint32-+0
	rel 48+8 t=R_ADDR type:[64]uint32+0
type:[64]uint32 SRODATA dupok size=72 align=0x8
	0x0000 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 85 e0 4f d7 0a 04 04 11 00 00 00 00 00 00 00 00  ..O.............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 40 00 00 00 00 00 00 00                          @.......
	rel 24+8 t=R_ADDR type:.eqfunc.M256+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[64]uint32-+0
	rel 44+4 t=RelocType(-32763) type:*[64]uint32+0
	rel 48+8 t=R_ADDR type:uint32+0
	rel 56+8 t=R_ADDR type:[]uint32+0
type:.namedata.*[][64]uint32- SRODATA dupok size=15 align=0x1
	0x0000 00 0d 2a 5b 5d 5b 36 34 5d 75 69 6e 74 33 32     ..*[][64]uint32
type:*[][64]uint32 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 0d d2 88 d5 28 08 08 16 00 00 00 00 00 00 00 00  ....(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[][64]uint32-+0
	rel 48+8 t=R_ADDR type:[][64]uint32+0
type:[][64]uint32 SRODATA dupok size=56 align=0x8
	0x0000 18 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 88 67 33 b8 02 08 08 17 00 00 00 00 00 00 00 00  .g3.............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[][64]uint32-+0
	rel 44+4 t=RelocType(-32763) type:*[][64]uint32+0
	rel 48+8 t=R_ADDR type:[64]uint32+0
type:[8][64]uint32 SRODATA dupok size=72 align=0x8
	0x0000 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 c8 37 7c b7 0a 04 04 11 00 00 00 00 00 00 00 00  .7|.............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 08 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR type:.eqfunc.M2048+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[8][64]uint32-+0
	rel 44+4 t=RelocType(-32763) type:*[8][64]uint32+0
	rel 48+8 t=R_ADDR type:[64]uint32+0
	rel 56+8 t=R_ADDR type:[][64]uint32+0
crypto/des.feistelBoxOnce SNOPTRBSS size=12 align=0x0
 SDWARFVAR size=42 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 66 65 69 73  .crypto/des.feis
	0x0010 74 65 6c 42 6f 78 4f 6e 63 65 00 09 03 00 00 00  telBoxOnce......
	0x0020 00 00 00 00 00 00 00 00 00 01                    ..........
	rel 29+8 t=R_ADDR crypto/des.feistelBoxOnce+0
	rel 37+4 t=R_DWARFSECREF go:info.sync.Once+0
go:constinfo.crypto/des SDWARFCONST size=27 align=0x0
	0x0000 0b 63 72 79 70 74 6f 2f 64 65 73 2e 42 6c 6f 63  .crypto/des.Bloc
	0x0010 6b 53 69 7a 65 00 00 00 00 00 08                 kSize......
	rel 22+4 t=R_DWARFSECREF go:info.int+0
crypto/des.initialPermutation SNOPTRDATA size=64 align=0x0
	0x0000 06 0e 16 1e 26 2e 36 3e 04 0c 14 1c 24 2c 34 3c  ....&.6>....$,4<
	0x0010 02 0a 12 1a 22 2a 32 3a 00 08 10 18 20 28 30 38  ...."*2:.... (08
	0x0020 07 0f 17 1f 27 2f 37 3f 05 0d 15 1d 25 2d 35 3d  ....'/7?....%-5=
	0x0030 03 0b 13 1b 23 2b 33 3b 01 09 11 19 21 29 31 39  ....#+3;....!)19
 SDWARFVAR size=46 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 69 6e 69 74  .crypto/des.init
	0x0010 69 61 6c 50 65 72 6d 75 74 61 74 69 6f 6e 00 09  ialPermutation..
	0x0020 03 00 00 00 00 00 00 00 00 00 00 00 00 01        ..............
	rel 33+8 t=R_ADDR crypto/des.initialPermutation+0
	rel 41+4 t=R_DWARFSECREF go:info.[64]uint8+0
type:.eqfunc.M64 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00  ........@.......
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.namedata.*[64]uint8- SRODATA dupok size=12 align=0x1
	0x0000 00 0a 2a 5b 36 34 5d 75 69 6e 74 38              ..*[64]uint8
type:*[64]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 5b 3a e6 82 28 08 08 16 00 00 00 00 00 00 00 00  [:..(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[64]uint8-+0
	rel 48+8 t=R_ADDR type:[64]uint8+0
type:[64]uint8 SRODATA dupok size=72 align=0x8
	0x0000 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  @...............
	0x0010 7f 99 24 3f 0a 01 01 11 00 00 00 00 00 00 00 00  ..$?............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 40 00 00 00 00 00 00 00                          @.......
	rel 24+8 t=R_ADDR type:.eqfunc.M64+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[64]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[64]uint8+0
	rel 48+8 t=R_ADDR type:uint8+0
	rel 56+8 t=R_ADDR type:[]uint8+0
crypto/des.finalPermutation SNOPTRDATA size=64 align=0x0
	0x0000 18 38 10 30 08 28 00 20 19 39 11 31 09 29 01 21  .8.0.(. .9.1.).!
	0x0010 1a 3a 12 32 0a 2a 02 22 1b 3b 13 33 0b 2b 03 23  .:.2.*.".;.3.+.#
	0x0020 1c 3c 14 34 0c 2c 04 24 1d 3d 15 35 0d 2d 05 25  .<.4.,.$.=.5.-.%
	0x0030 1e 3e 16 36 0e 2e 06 26 1f 3f 17 37 0f 2f 07 27  .>.6...&.?.7./.'
 SDWARFVAR size=44 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 66 69 6e 61  .crypto/des.fina
	0x0010 6c 50 65 72 6d 75 74 61 74 69 6f 6e 00 09 03 00  lPermutation....
	0x0020 00 00 00 00 00 00 00 00 00 00 00 01              ............
	rel 31+8 t=R_ADDR crypto/des.finalPermutation+0
	rel 39+4 t=R_DWARFSECREF go:info.[64]uint8+0
crypto/des.expansionFunction SNOPTRDATA size=48 align=0x0
	0x0000 00 1f 1e 1d 1c 1b 1c 1b 1a 19 18 17 18 17 16 15  ................
	0x0010 14 13 14 13 12 11 10 0f 10 0f 0e 0d 0c 0b 0c 0b  ................
	0x0020 0a 09 08 07 08 07 06 05 04 03 04 03 02 01 00 1f  ................
 SDWARFVAR size=45 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 65 78 70 61  .crypto/des.expa
	0x0010 6e 73 69 6f 6e 46 75 6e 63 74 69 6f 6e 00 09 03  nsionFunction...
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 01           .............
	rel 32+8 t=R_ADDR crypto/des.expansionFunction+0
	rel 40+4 t=R_DWARFSECREF go:info.[48]uint8+0
type:.eqfunc.M48 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 30 00 00 00 00 00 00 00  ........0.......
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.namedata.*[48]uint8- SRODATA dupok size=12 align=0x1
	0x0000 00 0a 2a 5b 34 38 5d 75 69 6e 74 38              ..*[48]uint8
type:*[48]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 13 38 5f 51 28 08 08 16 00 00 00 00 00 00 00 00  .8_Q(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[48]uint8-+0
	rel 48+8 t=R_ADDR type:[48]uint8+0
type:[48]uint8 SRODATA dupok size=72 align=0x8
	0x0000 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  0...............
	0x0010 83 8d a1 3c 0a 01 01 11 00 00 00 00 00 00 00 00  ...<............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 30 00 00 00 00 00 00 00                          0.......
	rel 24+8 t=R_ADDR type:.eqfunc.M48+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[48]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[48]uint8+0
	rel 48+8 t=R_ADDR type:uint8+0
	rel 56+8 t=R_ADDR type:[]uint8+0
crypto/des.permutationFunction SNOPTRDATA size=32 align=0x0
	0x0000 10 19 0c 0b 03 14 04 0f 1f 11 09 06 1b 0e 01 16  ................
	0x0010 1e 18 08 12 00 05 1d 17 0d 13 02 1a 0a 15 1c 07  ................
 SDWARFVAR size=47 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 70 65 72 6d  .crypto/des.perm
	0x0010 75 74 61 74 69 6f 6e 46 75 6e 63 74 69 6f 6e 00  utationFunction.
	0x0020 09 03 00 00 00 00 00 00 00 00 00 00 00 00 01     ...............
	rel 34+8 t=R_ADDR crypto/des.permutationFunction+0
	rel 42+4 t=R_DWARFSECREF go:info.[32]uint8+0
type:.eqfunc.M32 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00  ........ .......
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.namedata.*[32]uint8- SRODATA dupok size=12 align=0x1
	0x0000 00 0a 2a 5b 33 32 5d 75 69 6e 74 38              ..*[32]uint8
type:*[32]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 26 f2 4c 61 28 08 08 16 00 00 00 00 00 00 00 00  &.La(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[32]uint8-+0
	rel 48+8 t=R_ADDR type:[32]uint8+0
type:[32]uint8 SRODATA dupok size=72 align=0x8
	0x0000 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00   ...............
	0x0010 aa 54 6c 6d 0a 01 01 11 00 00 00 00 00 00 00 00  .Tlm............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 20 00 00 00 00 00 00 00                           .......
	rel 24+8 t=R_ADDR type:.eqfunc.M32+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[32]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[32]uint8+0
	rel 48+8 t=R_ADDR type:uint8+0
	rel 56+8 t=R_ADDR type:[]uint8+0
crypto/des.permutedChoice1 SNOPTRDATA size=56 align=0x0
	0x0000 07 0f 17 1f 27 2f 37 3f 06 0e 16 1e 26 2e 36 3e  ....'/7?....&.6>
	0x0010 05 0d 15 1d 25 2d 35 3d 04 0c 14 1c 01 09 11 19  ....%-5=........
	0x0020 21 29 31 39 02 0a 12 1a 22 2a 32 3a 03 0b 13 1b  !)19...."*2:....
	0x0030 23 2b 33 3b 24 2c 34 3c                          #+3;$,4<
 SDWARFVAR size=43 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 70 65 72 6d  .crypto/des.perm
	0x0010 75 74 65 64 43 68 6f 69 63 65 31 00 09 03 00 00  utedChoice1.....
	0x0020 00 00 00 00 00 00 00 00 00 00 01                 ...........
	rel 30+8 t=R_ADDR crypto/des.permutedChoice1+0
	rel 38+4 t=R_DWARFSECREF go:info.[56]uint8+0
type:.eqfunc.M56 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 38 00 00 00 00 00 00 00  ........8.......
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.namedata.*[56]uint8- SRODATA dupok size=12 align=0x1
	0x0000 00 0a 2a 5b 35 36 5d 75 69 6e 74 38              ..*[56]uint8
type:*[56]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 b0 64 eb be 28 08 08 16 00 00 00 00 00 00 00 00  .d..(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[56]uint8-+0
	rel 48+8 t=R_ADDR type:[56]uint8+0
type:[56]uint8 SRODATA dupok size=72 align=0x8
	0x0000 38 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  8...............
	0x0010 fc cd 60 75 0a 01 01 11 00 00 00 00 00 00 00 00  ..`u............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 38 00 00 00 00 00 00 00                          8.......
	rel 24+8 t=R_ADDR type:.eqfunc.M56+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[56]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[56]uint8+0
	rel 48+8 t=R_ADDR type:uint8+0
	rel 56+8 t=R_ADDR type:[]uint8+0
crypto/des.permutedChoice2 SNOPTRDATA size=48 align=0x0
	0x0000 2a 27 2d 20 37 33 35 1c 29 32 23 2e 21 25 2c 34  *'- 735.)2#.!%,4
	0x0010 1e 30 28 31 1d 24 2b 36 0f 04 19 13 09 01 1a 10  .0(1.$+6........
	0x0020 05 0b 17 08 0c 07 11 00 16 03 0a 0e 06 14 1b 18  ................
 SDWARFVAR size=43 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 70 65 72 6d  .crypto/des.perm
	0x0010 75 74 65 64 43 68 6f 69 63 65 32 00 09 03 00 00  utedChoice2.....
	0x0020 00 00 00 00 00 00 00 00 00 00 01                 ...........
	rel 30+8 t=R_ADDR crypto/des.permutedChoice2+0
	rel 38+4 t=R_DWARFSECREF go:info.[48]uint8+0
crypto/des.sBoxes SNOPTRDATA size=512 align=0x0
	0x0000 0e 04 0d 01 02 0f 0b 08 03 0a 06 0c 05 09 00 07  ................
	0x0010 00 0f 07 04 0e 02 0d 01 0a 06 0c 0b 09 05 03 08  ................
	0x0020 04 01 0e 08 0d 06 02 0b 0f 0c 09 07 03 0a 05 00  ................
	0x0030 0f 0c 08 02 04 09 01 07 05 0b 03 0e 0a 00 06 0d  ................
	0x0040 0f 01 08 0e 06 0b 03 04 09 07 02 0d 0c 00 05 0a  ................
	0x0050 03 0d 04 07 0f 02 08 0e 0c 00 01 0a 06 09 0b 05  ................
	0x0060 00 0e 07 0b 0a 04 0d 01 05 08 0c 06 09 03 02 0f  ................
	0x0070 0d 08 0a 01 03 0f 04 02 0b 06 07 0c 00 05 0e 09  ................
	0x0080 0a 00 09 0e 06 03 0f 05 01 0d 0c 07 0b 04 02 08  ................
	0x0090 0d 07 00 09 03 04 06 0a 02 08 05 0e 0c 0b 0f 01  ................
	0x00a0 0d 06 04 09 08 0f 03 00 0b 01 02 0c 05 0a 0e 07  ................
	0x00b0 01 0a 0d 00 06 09 08 07 04 0f 0e 03 0b 05 02 0c  ................
	0x00c0 07 0d 0e 03 00 06 09 0a 01 02 08 05 0b 0c 04 0f  ................
	0x00d0 0d 08 0b 05 06 0f 00 03 04 07 02 0c 01 0a 0e 09  ................
	0x00e0 0a 06 09 00 0c 0b 07 0d 0f 01 03 0e 05 02 08 04  ................
	0x00f0 03 0f 00 06 0a 01 0d 08 09 04 05 0b 0c 07 02 0e  ................
	0x0100 02 0c 04 01 07 0a 0b 06 08 05 03 0f 0d 00 0e 09  ................
	0x0110 0e 0b 02 0c 04 07 0d 01 05 00 0f 0a 03 09 08 06  ................
	0x0120 04 02 01 0b 0a 0d 07 08 0f 09 0c 05 06 03 00 0e  ................
	0x0130 0b 08 0c 07 01 0e 02 0d 06 0f 00 09 0a 04 05 03  ................
	0x0140 0c 01 0a 0f 09 02 06 08 00 0d 03 04 0e 07 05 0b  ................
	0x0150 0a 0f 04 02 07 0c 09 05 06 01 0d 0e 00 0b 03 08  ................
	0x0160 09 0e 0f 05 02 08 0c 03 07 00 04 0a 01 0d 0b 06  ................
	0x0170 04 03 02 0c 09 05 0f 0a 0b 0e 01 07 06 00 08 0d  ................
	0x0180 04 0b 02 0e 0f 00 08 0d 03 0c 09 07 05 0a 06 01  ................
	0x0190 0d 00 0b 07 04 09 01 0a 0e 03 05 0c 02 0f 08 06  ................
	0x01a0 01 04 0b 0d 0c 03 07 0e 0a 0f 06 08 00 05 09 02  ................
	0x01b0 06 0b 0d 08 01 04 0a 07 09 05 00 0f 0e 02 03 0c  ................
	0x01c0 0d 02 08 04 06 0f 0b 01 0a 09 03 0e 05 00 0c 07  ................
	0x01d0 01 0f 0d 08 0a 03 07 04 0c 05 06 0b 00 0e 09 02  ................
	0x01e0 07 0b 04 01 09 0c 0e 02 00 06 0a 0d 0f 03 05 08  ................
	0x01f0 02 01 0e 07 04 0a 08 0d 0f 0c 09 00 03 05 06 0b  ................
 SDWARFVAR size=34 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 73 42 6f 78  .crypto/des.sBox
	0x0010 65 73 00 09 03 00 00 00 00 00 00 00 00 00 00 00  es..............
	0x0020 00 01                                            ..
	rel 21+8 t=R_ADDR crypto/des.sBoxes+0
	rel 29+4 t=R_DWARFSECREF go:info.[8][4][16]uint8+0
type:.eqfunc.M512 SRODATA dupok size=16 align=0x0
	0x0000 00 00 00 00 00 00 00 00 00 02 00 00 00 00 00 00  ................
	rel 0+8 t=R_ADDR runtime.memequal_varlen+0
type:.namedata.*[8][4][16]uint8- SRODATA dupok size=18 align=0x1
	0x0000 00 10 2a 5b 38 5d 5b 34 5d 5b 31 36 5d 75 69 6e  ..*[8][4][16]uin
	0x0010 74 38                                            t8
type:*[8][4][16]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 4a 3c fe 9b 28 08 08 16 00 00 00 00 00 00 00 00  J<..(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[8][4][16]uint8-+0
	rel 48+8 t=R_ADDR type:[8][4][16]uint8+0
type:.namedata.*[4][16]uint8- SRODATA dupok size=15 align=0x1
	0x0000 00 0d 2a 5b 34 5d 5b 31 36 5d 75 69 6e 74 38     ..*[4][16]uint8
type:*[4][16]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 b5 f0 2e 01 28 08 08 16 00 00 00 00 00 00 00 00  ....(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[4][16]uint8-+0
	rel 48+8 t=R_ADDR type:[4][16]uint8+0
runtime.memequal128·f SRODATA dupok size=8 align=0x0
	0x0000 00 00 00 00 00 00 00 00                          ........
	rel 0+8 t=R_ADDR runtime.memequal128+0
type:.namedata.*[16]uint8- SRODATA dupok size=12 align=0x1
	0x0000 00 0a 2a 5b 31 36 5d 75 69 6e 74 38              ..*[16]uint8
type:*[16]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 27 5d bc 2f 28 08 08 16 00 00 00 00 00 00 00 00  ']./(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[16]uint8-+0
	rel 48+8 t=R_ADDR type:[16]uint8+0
type:[16]uint8 SRODATA dupok size=72 align=0x8
	0x0000 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 aa eb d0 49 0a 01 01 11 00 00 00 00 00 00 00 00  ...I............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 10 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal128·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[16]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[16]uint8+0
	rel 48+8 t=R_ADDR type:uint8+0
	rel 56+8 t=R_ADDR type:[]uint8+0
type:.namedata.*[][16]uint8- SRODATA dupok size=14 align=0x1
	0x0000 00 0c 2a 5b 5d 5b 31 36 5d 75 69 6e 74 38        ..*[][16]uint8
type:*[][16]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 1e 61 49 0e 28 08 08 16 00 00 00 00 00 00 00 00  .aI.(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[][16]uint8-+0
	rel 48+8 t=R_ADDR type:[][16]uint8+0
type:[][16]uint8 SRODATA dupok size=56 align=0x8
	0x0000 18 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 7b 0d 3f e1 02 08 08 17 00 00 00 00 00 00 00 00  {.?.............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[][16]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[][16]uint8+0
	rel 48+8 t=R_ADDR type:[16]uint8+0
type:[4][16]uint8 SRODATA dupok size=72 align=0x8
	0x0000 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  @...............
	0x0010 fc fc 29 ae 0a 01 01 11 00 00 00 00 00 00 00 00  ..).............
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 04 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR type:.eqfunc.M64+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[4][16]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[4][16]uint8+0
	rel 48+8 t=R_ADDR type:[16]uint8+0
	rel 56+8 t=R_ADDR type:[][16]uint8+0
type:.namedata.*[][4][16]uint8- SRODATA dupok size=17 align=0x1
	0x0000 00 0f 2a 5b 5d 5b 34 5d 5b 31 36 5d 75 69 6e 74  ..*[][4][16]uint
	0x0010 38                                               8
type:*[][4][16]uint8 SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 f0 3f 9b bf 28 08 08 16 00 00 00 00 00 00 00 00  .?..(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[][4][16]uint8-+0
	rel 48+8 t=R_ADDR type:[][4][16]uint8+0
type:[][4][16]uint8 SRODATA dupok size=56 align=0x8
	0x0000 18 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 f7 db 87 ff 02 08 08 17 00 00 00 00 00 00 00 00  ................
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[][4][16]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[][4][16]uint8+0
	rel 48+8 t=R_ADDR type:[4][16]uint8+0
type:[8][4][16]uint8 SRODATA dupok size=72 align=0x8
	0x0000 00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 b0 d3 c6 9d 0a 01 01 11 00 00 00 00 00 00 00 00  ................
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0040 08 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR type:.eqfunc.M512+0
	rel 32+8 t=R_ADDR runtime.gcbits.+0
	rel 40+4 t=R_ADDROFF type:.namedata.*[8][4][16]uint8-+0
	rel 44+4 t=RelocType(-32763) type:*[8][4][16]uint8+0
	rel 48+8 t=R_ADDR type:[4][16]uint8+0
	rel 56+8 t=R_ADDR type:[][4][16]uint8+0
crypto/des.ksRotations SNOPTRDATA size=16 align=0x0
	0x0000 01 01 02 02 02 02 02 02 01 02 02 02 02 02 02 01  ................
 SDWARFVAR size=39 align=0x0
	0x0000 0a 63 72 79 70 74 6f 2f 64 65 73 2e 6b 73 52 6f  .crypto/des.ksRo
	0x0010 74 61 74 69 6f 6e 73 00 09 03 00 00 00 00 00 00  tations.........
	0x0020 00 00 00 00 00 00 01                             .......
	rel 26+8 t=R_ADDR crypto/des.ksRotations+0
	rel 34+4 t=R_DWARFSECREF go:info.[16]uint8+0
go:string."crypto/des: invalid key size " SRODATA dupok size=29 align=0x1
	0x0000 63 72 79 70 74 6f 2f 64 65 73 3a 20 69 6e 76 61  crypto/des: inva
	0x0010 6c 69 64 20 6b 65 79 20 73 69 7a 65 20           lid key size 
go:string."crypto/des: use of DES is not allowed in FIPS 140-only mode" SRODATA dupok size=59 align=0x1
	0x0000 63 72 79 70 74 6f 2f 64 65 73 3a 20 75 73 65 20  crypto/des: use 
	0x0010 6f 66 20 44 45 53 20 69 73 20 6e 6f 74 20 61 6c  of DES is not al
	0x0020 6c 6f 77 65 64 20 69 6e 20 46 49 50 53 20 31 34  lowed in FIPS 14
	0x0030 30 2d 6f 6e 6c 79 20 6d 6f 64 65                 0-only mode
go:string."crypto/des: input not full block" SRODATA dupok size=32 align=0x1
	0x0000 63 72 79 70 74 6f 2f 64 65 73 3a 20 69 6e 70 75  crypto/des: inpu
	0x0010 74 20 6e 6f 74 20 66 75 6c 6c 20 62 6c 6f 63 6b  t not full block
go:string."crypto/des: output not full block" SRODATA dupok size=33 align=0x1
	0x0000 63 72 79 70 74 6f 2f 64 65 73 3a 20 6f 75 74 70  crypto/des: outp
	0x0010 75 74 20 6e 6f 74 20 66 75 6c 6c 20 62 6c 6f 63  ut not full bloc
	0x0020 6b                                               k
go:string."crypto/des: invalid buffer overlap" SRODATA dupok size=34 align=0x1
	0x0000 63 72 79 70 74 6f 2f 64 65 73 3a 20 69 6e 76 61  crypto/des: inva
	0x0010 6c 69 64 20 62 75 66 66 65 72 20 6f 76 65 72 6c  lid buffer overl
	0x0020 61 70                                            ap
crypto/des..stmp_0 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00  ........ .......
	rel 0+8 t=R_ADDR go:string."crypto/des: input not full block"+0
crypto/des..stmp_1 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00  ........!.......
	rel 0+8 t=R_ADDR go:string."crypto/des: output not full block"+0
crypto/des..stmp_2 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 22 00 00 00 00 00 00 00  ........".......
	rel 0+8 t=R_ADDR go:string."crypto/des: invalid buffer overlap"+0
crypto/des..stmp_3 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00  ........ .......
	rel 0+8 t=R_ADDR go:string."crypto/des: input not full block"+0
crypto/des..stmp_4 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00  ........!.......
	rel 0+8 t=R_ADDR go:string."crypto/des: output not full block"+0
crypto/des..stmp_5 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 22 00 00 00 00 00 00 00  ........".......
	rel 0+8 t=R_ADDR go:string."crypto/des: invalid buffer overlap"+0
go:string."crypto/des: use of TripleDES is not allowed in FIPS 140-only mode" SRODATA dupok size=65 align=0x1
	0x0000 63 72 79 70 74 6f 2f 64 65 73 3a 20 75 73 65 20  crypto/des: use 
	0x0010 6f 66 20 54 72 69 70 6c 65 44 45 53 20 69 73 20  of TripleDES is 
	0x0020 6e 6f 74 20 61 6c 6c 6f 77 65 64 20 69 6e 20 46  not allowed in F
	0x0030 49 50 53 20 31 34 30 2d 6f 6e 6c 79 20 6d 6f 64  IPS 140-only mod
	0x0040 65                                               e
crypto/des..stmp_6 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00  ........ .......
	rel 0+8 t=R_ADDR go:string."crypto/des: input not full block"+0
crypto/des..stmp_7 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00  ........!.......
	rel 0+8 t=R_ADDR go:string."crypto/des: output not full block"+0
crypto/des..stmp_8 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 22 00 00 00 00 00 00 00  ........".......
	rel 0+8 t=R_ADDR go:string."crypto/des: invalid buffer overlap"+0
crypto/des..stmp_9 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00  ........ .......
	rel 0+8 t=R_ADDR go:string."crypto/des: input not full block"+0
crypto/des..stmp_10 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 21 00 00 00 00 00 00 00  ........!.......
	rel 0+8 t=R_ADDR go:string."crypto/des: output not full block"+0
crypto/des..stmp_11 SRODATA static size=16 align=0x8
	0x0000 00 00 00 00 00 00 00 00 22 00 00 00 00 00 00 00  ........".......
	rel 0+8 t=R_ADDR go:string."crypto/des: invalid buffer overlap"+0
type:.namedata.*func()- SRODATA dupok size=9 align=0x1
	0x0000 00 07 2a 66 75 6e 63 28 29                       ..*func()
type:*func() SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 75 53 d6 d8 28 08 08 16 00 00 00 00 00 00 00 00  uS..(...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 00                          ........
	rel 24+8 t=R_ADDR runtime.memequal64·f+0
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func()-+0
	rel 48+8 t=R_ADDR type:func()+0
type:func() SRODATA dupok size=56 align=0x8
	0x0000 08 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00  ................
	0x0010 fe 05 46 7f 22 08 08 13 00 00 00 00 00 00 00 00  ..F."...........
	0x0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0030 00 00 00 00                                      ....
	rel 32+8 t=R_ADDR runtime.gcbits.0100000000000000+0
	rel 40+4 t=R_ADDROFF type:.namedata.*func()-+0
	rel 44+4 t=RelocType(-32763) type:*func()+0
crypto/des.initFeistelBox·f SRODATA dupok size=8 align=0x0
	0x0000 00 00 00 00 00 00 00 00                          ........
	rel 0+8 t=R_ADDR crypto/des.initFeistelBox+0
gclocals·g5+hNtRBP6YXNjfog7aZjQ== SRODATA dupok size=8 align=0x4
	0x0000 01 00 00 00 00 00 00 00                          ........
gclocals·h6wMYRdCOACbyO8hHOhGLw== SRODATA dupok size=12 align=0x4
	0x0000 04 00 00 00 08 00 00 00 92 12 10 00              ............
gclocals·D6AnjkNyRZzKYVnNXnHP7g== SRODATA dupok size=8 align=0x4
	0x0000 04 00 00 00 00 00 00 00                          ........
crypto/des.cryptBlock.arginfo1 SRODATA static dupok size=27 align=0x1
	0x0000 fe 08 08 10 08 18 08 fd fe 20 08 28 08 30 08 fd  ......... .(.0..
	0x0010 fe 38 08 40 08 48 08 fd 00 01 ff                 .8.@.H.....
crypto/des.cryptBlock.argliveinfo SRODATA static dupok size=5 align=0x1
	0x0000 08 00 00 1b 00                                   .....
crypto/des.feistel.arginfo1 SRODATA static dupok size=9 align=0x1
	0x0000 00 04 04 04 08 08 10 08 ff                       .........
crypto/des.feistel.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
gclocals·Ih7UaEzxol4qYE7mnpMg6w== SRODATA dupok size=10 align=0x4
	0x0000 02 00 00 00 02 00 00 00 02 00                    ..........
gclocals·J26BEvPExEQhJvjp9E8Whg== SRODATA dupok size=8 align=0x4
	0x0000 02 00 00 00 00 00 00 00                          ........
crypto/des.permuteBlock.arginfo1 SRODATA static dupok size=11 align=0x1
	0x0000 00 08 fe 08 08 10 08 18 08 fd ff                 ...........
crypto/des.permuteBlock.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
crypto/des.permuteInitialBlock.arginfo1 SRODATA static dupok size=3 align=0x1
	0x0000 00 08 ff                                         ...
crypto/des.permuteInitialBlock.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
crypto/des.permuteFinalBlock.arginfo1 SRODATA static dupok size=3 align=0x1
	0x0000 00 08 ff                                         ...
crypto/des.permuteFinalBlock.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
crypto/des.ksRotate.arginfo1 SRODATA static dupok size=3 align=0x1
	0x0000 00 04 ff                                         ...
crypto/des.ksRotate.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
gclocals·TswRR9Pia9Wsluv5u1sUnA== SRODATA dupok size=10 align=0x4
	0x0000 02 00 00 00 02 00 00 00 03 00                    ..........
crypto/des.(*desCipher).generateSubkeys.arginfo1 SRODATA static dupok size=11 align=0x1
	0x0000 00 08 fe 08 08 10 08 18 08 fd ff                 ...........
crypto/des.(*desCipher).generateSubkeys.argliveinfo SRODATA static dupok size=3 align=0x1
	0x0000 00 00 07                                         ...
crypto/des.unpack.arginfo1 SRODATA static dupok size=3 align=0x1
	0x0000 00 08 ff                                         ...
crypto/des.unpack.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
crypto/des.KeySizeError.Error.arginfo1 SRODATA static dupok size=3 align=0x1
	0x0000 00 08 ff                                         ...
crypto/des.KeySizeError.Error.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
gclocals·Z8zdw/dq+fE82FieA9ctlQ== SRODATA dupok size=11 align=0x4
	0x0000 03 00 00 00 01 00 00 00 01 00 00                 ...........
gclocals·dJUzpsp3Yf15gqnc4dLliw== SRODATA dupok size=11 align=0x4
	0x0000 03 00 00 00 01 00 00 00 00 00 01                 ...........
crypto/des.NewCipher.arginfo1 SRODATA static dupok size=9 align=0x1
	0x0000 fe 00 08 08 08 10 08 fd ff                       .........
crypto/des.NewCipher.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
gclocals·PqVQJggGJ2Sanl8iJuPOkQ== SRODATA dupok size=9 align=0x4
	0x0000 01 00 00 00 01 00 00 00 00                       .........
crypto/des.(*desCipher).BlockSize.arginfo1 SRODATA static dupok size=3 align=0x1
	0x0000 00 08 ff                                         ...
crypto/des.(*desCipher).BlockSize.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
gclocals·rZc36ENI8UsaczlPbgMejw== SRODATA dupok size=10 align=0x4
	0x0000 02 00 00 00 05 00 00 00 13 00                    ..........
crypto/des.(*desCipher).Encrypt.arginfo1 SRODATA static dupok size=19 align=0x1
	0x0000 00 08 fe 08 08 10 08 18 08 fd fe 20 08 28 08 30  ........... .(.0
	0x0010 08 fd ff                                         ...
crypto/des.(*desCipher).Encrypt.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
crypto/des.(*desCipher).Decrypt.arginfo1 SRODATA static dupok size=19 align=0x1
	0x0000 00 08 fe 08 08 10 08 18 08 fd fe 20 08 28 08 30  ........... .(.0
	0x0010 08 fd ff                                         ...
crypto/des.(*desCipher).Decrypt.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
gclocals·jqXdKVKXqQEE0nK8rJ96oA== SRODATA dupok size=12 align=0x4
	0x0000 04 00 00 00 01 00 00 00 01 00 01 00              ............
gclocals·yjRPzIZTs1w23N5U7zM9ew== SRODATA dupok size=12 align=0x4
	0x0000 04 00 00 00 01 00 00 00 00 00 01 01              ............
crypto/des.NewTripleDESCipher.arginfo1 SRODATA static dupok size=9 align=0x1
	0x0000 fe 00 08 08 08 10 08 fd ff                       .........
crypto/des.NewTripleDESCipher.argliveinfo SRODATA static dupok size=3 align=0x1
	0x0000 00 00 05                                         ...
crypto/des.(*tripleDESCipher).BlockSize.arginfo1 SRODATA static dupok size=3 align=0x1
	0x0000 00 08 ff                                         ...
crypto/des.(*tripleDESCipher).BlockSize.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
gclocals·+CyRcNapSOQVXQOoLfXQCQ== SRODATA dupok size=12 align=0x4
	0x0000 04 00 00 00 05 00 00 00 13 03 02 00              ............
crypto/des.(*tripleDESCipher).Encrypt.arginfo1 SRODATA static dupok size=19 align=0x1
	0x0000 00 08 fe 08 08 10 08 18 08 fd fe 20 08 28 08 30  ........... .(.0
	0x0010 08 fd ff                                         ...
crypto/des.(*tripleDESCipher).Encrypt.argliveinfo SRODATA static dupok size=3 align=0x1
	0x0000 00 00 03                                         ...
crypto/des.(*tripleDESCipher).Decrypt.arginfo1 SRODATA static dupok size=19 align=0x1
	0x0000 00 08 fe 08 08 10 08 18 08 fd fe 20 08 28 08 30  ........... .(.0
	0x0010 08 fd ff                                         ...
crypto/des.(*tripleDESCipher).Decrypt.argliveinfo SRODATA static dupok size=3 align=0x1
	0x0000 00 00 03                                         ...
gclocals·wvjpxkknJ4nY1JtrArJJaw== SRODATA dupok size=10 align=0x4
	0x0000 02 00 00 00 01 00 00 00 01 00                    ..........
crypto/des.(*KeySizeError).Error.arginfo1 SRODATA static dupok size=3 align=0x1
	0x0000 00 08 ff                                         ...
crypto/des.(*KeySizeError).Error.argliveinfo SRODATA static dupok size=2 align=0x1
	0x0000 00 00                                            ..
