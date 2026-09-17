#!/usr/bin/env python3
"""Project IBFV improvement as iris GAR increases."""

for label, ga, uni_gacc, ibfv_gacc in [
    ('DB3', 95, 87, 92),
    ('DB4', 100, 91, 98),
]:
    uni_frr = (ga - uni_gacc) / ga * 100
    ibfv_frr_54 = (ga - ibfv_gacc) / ga * 100
    rescued_at_54 = ibfv_gacc - uni_gacc
    uni_rejected = ga - uni_gacc

    print(f'{label}: {ga} genuine attempts')
    print(f'  Unimodal: {uni_gacc}/{ga} accept (FRR={uni_frr:.1f}%)')
    print(f'  IBFV@54%: {ibfv_gacc}/{ga} accept (FRR={ibfv_frr_54:.1f}%)')
    print(f'  Rescued by iris: {rescued_at_54}/{uni_rejected} of rejected users')
    
    rescue_rate = min(rescued_at_54 / (0.54 * uni_rejected), 1.0)
    print(f'  Rescue rate (when iris succeeds): {rescue_rate*100:.1f}%')
    print()
    print(f'  {"GAR":>6} {"Rescued":>8} {"FRR":>8} {"EER~":>8}')
    print(f'  {"-"*6} {"-"*8} {"-"*8} {"-"*8}')
    
    for gar in [0.54, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95]:
        new_rescued = min(gar * uni_rejected * rescue_rate, uni_rejected)
        new_gacc = uni_gacc + new_rescued
        new_frr = (ga - new_gacc) / ga * 100
        new_eer = new_frr / 2
        marker = " ← current" if gar == 0.54 else ""
        print(f'  {gar:>5.0%} {new_rescued:>8.1f} {new_frr:>7.2f}% {new_eer:>7.2f}%{marker}')
    print()
