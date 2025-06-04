Based on [OpenCheetah](https://github.com/Alibaba-Gemini-Lab/OpenCheetah/tree/main)


# Protocols

# Protocol 1

Step 1:
- Party 1: A1' = Enc(A1), B1' = Enc(B1), send A1', B1' to Party 2

Step 2:
- Party 2: M = A1'_B2 + A2_B1' - R, send M to Party 1

Step 3:
- Party 1: C1 = A1*B1 + Dec(M)
- Party 2: C2 = A2*B2 + R

## Protocol 2

Step 1:
- Party 1: A1' = Enc(A1), send A1' to Party 2:
- Party 2: A2' = Enc(A2), send A2' to Party 1

Step 2:
- Party 1: M1 = A2' B1 + R1, send M1 to Party 2
- Party 2: M2 = A1' B2 + R2, send M2 to Party 1

Step 2:
- Party 1: C1 = A1*B1 + Dec(M2) - R1
- Party 2: C2 = A2*B2 + DEC(M1) - R2

## Protocol 3

Step 1:
- Party 1: A1' = Enc(A1), send A1' to Party 2

Step 2:
- Party 2: M2 = (A1' +A2) B2 - R2, send M2 to Party 1

Step 3:
- Party 1: C1 = Dec(M2) 
- Party 2: C2 = R2