# Simple-Medical-Chatting-based-on-RAG
This is a RAG (Retrieval-Augmented Generation) system built on simple online database of medical therapy
---
## Demo
### Example 1: Headache caused by staying up late
**User:**  
I have a headache and stayed up late studying recently.

**Assistant:**  
Since you often stay up late studying and have an irregular schedule, your condition is likely a tension headache.

**Treatment plan**
- **Immediate relief**  
For tension headaches: gently squeeze the base of your middle finger with a rubber band for 10 seconds to relieve pain.

- **Long-term prevention**  
Avoid working under fluorescent lights for too long; use warm desk lamps to reduce blue light interference and improve sleep.

Since your condition is mild, no medication is needed.

---

### Example 2: Headache with stuffy nose
**User:**  
I have a headache and a stuffy nose.

**Assistant:**  
You have both pain and nasal congestion, which suggests you may have a common cold or mild respiratory illness.

**Treatment plan**
- **Home remedies**  
You can make anti-cold tea: 5g honeysuckle + 3g mint + 2g licorice.  
Get 15 minutes of sunlight daily to boost immunity.

- **Environmental care**  
Keep your room clean and ventilated to reduce viruses and allergens.

- **Medication**  
You may take Paracetamol or Ammonia Kahuang Min Capsules.

**Warnings:**  
Do not drink alcohol while taking.  
Use with caution if you have high blood pressure.

---

### Example 3: Stomachache consultation
**User:**  
I have a stomachache.

**Assistant:**  
Your symptoms indicate a digestive system issue, such as abdominal pain, bloating, or mild gastroenteritis.

**Treatment plan**
- **Non-medication**  
Eat light, avoid spicy or greasy food; rest and stay hydrated.

- **Medication**  
You may take Montmorillonite Powder or Oral Rehydration Salts.

**Warnings:**  
If you have a high fever, dehydration, or bloody stool, please see a doctor immediately.


---
## Methodology
1. Collecting therapy information online
2. Loading and chunking a medical knowledge PDF
3. Building a vector database for fast retrieval
4. Choosing a fixed, pre-trained LLM (ChatTongyi)
5. Using prompt engineering to control output format
