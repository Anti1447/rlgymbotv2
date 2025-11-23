# RLGymBot_v2 Training Roadmap

This document provides a structured training roadmap for developing a high‑level Rocket League reinforcement learning agent using the RLGymBot_v2 project.

---

## Phase 1 — Ball Touching (Bronze 0)

**Learning Rate:** `2e-4`  
**Total Steps:** 20M–30M  
**Milestones:** 5M, 10M, 15M, 20M  

### Goal
- Learn to reliably touch the ball.
- Learn to jump without collapsing into only driving.

### Promotion Criteria
- Touch rate ~70–90%
- Jump usage remains stable

---

## Phase 2 — Shooting Introduction (Bronze → Silver)

**Learning Rate:** `2e-4` → `1.5e-4` mid‑phase  
**Total Steps:** 40M–60M  
**Milestones:** every 5M  

### Goal
- Learn to push or shoot the ball forward.
- Begin intentional goal‑directed touches.

### Promotion Criteria
- Bot attempts shots
- Scoring chance 5–15%
- No “touch farming” behavior

---

## Phase 3 — Silver Fundamentals

**Learning Rate:** `1e-4`  
**Total Steps:** 80M–120M  
**Milestones:** every 10M  

### Goal
- Strong purposeful hits
- Jump shots
- Boost efficiency
- Consistent shooting attempts

### Promotion Criteria
- Bot scores regularly
- Shows strong hits
- Attempts jump shots
- Boost usage improves

---

## Phase 4 — Gold Skills 1

**Learning Rate:** `8e-5`  
**Total Steps:** 80M–160M  
**Milestones:** every 10M  

### Goal
- Begin simple aerials
- Better saves
- Better boost management
- Shooting consistency

### Promotion Criteria
- Bot attempts basic aerial touches
- Bot challenges more effectively
- Boost patterns appear intentional

---

## Transition to Self‑Play

Only enable self‑play **after Phase 3** when the bot:

- Scores reliably  
- Performs jump shots  
- Hits with power  
- Uses boost intelligently  

Self‑play becomes the primary driver of improvement from Gold onward.

---

## Recommended Long‑Term Plan

### Platinum (Phase 5)
- Encourage power shots
- Better defensive shadowing
- Better challenge timing

### Diamond (Phase 6)
- Encourage light aerial control
- Introduce minimal flip reset shaping
- Reward advanced touches slightly

### Champion → Grand Champion (Phase 7+)
- Full self‑play with occasional scripted opponents
- ELO evaluation loop
- Remove most shaping rewards

---

End of Roadmap.
