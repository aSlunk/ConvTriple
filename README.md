Based on [OpenCheetah](https://github.com/Alibaba-Gemini-Lab/OpenCheetah/tree/main)

# Protocols

- Currently supported:
    - boolean triples via OT
    - Matrix $\times$ Vector triples via HE
    - Convolution triples via HE

## Protocol 1

> [!NOTE]
> Both parties have shares of the image (A1/A2) and the filters (B1/B2)
> - Party1:
>   - IN: $A1, B1$
>   - OUT: $C1 = M2 + R1$
> - Party2:
>   - IN: $A2, B2$
>   - OUT: $C2 = M1 + R2$
>
> Verify: $C1 + C2 = (A1 + A2) \odot (B1 + B2)$

Step 1:
- Party 1: A1' = Enc(A1), send A1' to Party 2
- Party 2: A2' = Enc(A2), send A2' to Party 1

Step 2:
- Party 1: M1 = (A1 + A2') B1 - R1, send M1 to Party 2
- Party 2: M2 = (A1' + A2) B2 - R2, send M2 to Party 1

Step 2:
- Party 1: C1 = Dec(M2) + R1
- Party 2: C2 = DEC(M1) + R2

## Protocol 2

> [!NOTE]
> Both parties have shares of the image (A1/A2) but only `Party 2` has the filters (B2)
> - Party1:
>   - IN: $A1$
>   - OUT: $C1 = M2$
> - Party2:
>   - IN: $A2, B2$
>   - OUT: $C2 = R2$
>
> Verify $C1 + C2 = (A1 + A2) \odot B2$

Step 1:
- Party 1: A1' = Enc(A1), send A1' to Party 2

Step 2:
- Party 2: M2 = (A1' + A2) B2 - R2, send M2 to Party 1

Step 3:
- Party 1: C1 = Dec(M2) 
- Party 2: C2 = R2
