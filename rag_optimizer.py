"""
RAG-Augmented LLM Optimizer
Dual-loop: TVB simulation (quantitative) + gaps.json retrieval (qualitative)
"""
import json
import re
import anthropic
from simulate_v2 import run_robust_clinical, get_labels

client = anthropic.Anthropic()
labels = get_labels()

# 加载知识库
with open("gaps.json") as f:
    papers = json.load(f)

# 构建检索语料：每篇论文的title + open_questions
knowledge_base = []
for p in papers:
    for q in p["gaps"].get("open_questions", []):
        knowledge_base.append({
            "pmid": p["pmid"],
            "title": p["title"],
            "question": q
        })
print(f"Knowledge base: {len(knowledge_base)} entries from {len(papers)} papers")

def retrieve_knowledge(site_name, iext_boost, last_reward_trend, k=3):
    """
    Keyword-based retrieval from gaps.json
    trend: 'improving', 'worsening', 'plateau'
    """
    keywords = []
    
    # 基于位点的关键词
    site_keywords = {
        "rHC": ["hippocampus", "hippocampal", "temporal lobe", "limbic"],
        "lHC": ["hippocampus", "hippocampal", "temporal lobe", "limbic"],
        "rFEF": ["frontal", "prefrontal", "motor cortex"],
        "rTCS": ["temporal", "cortex", "neocortical"],
    }
    keywords.extend(site_keywords.get(site_name, ["stimulation", "epilepsy"]))
    
    # 基于趋势的关键词
    if last_reward_trend == "worsening":
        keywords.extend(["paradoxical", "worsening", "adverse", "contraindication"])
    elif last_reward_trend == "plateau":
        keywords.extend(["frequency", "amplitude", "optimization", "parameter"])
    else:
        keywords.extend(["suppression", "inhibition", "efficacy"])
    
    # 基于强度的关键词
    if iext_boost > 2.5:
        keywords.extend(["high frequency", "depolarization block", "safety"])
    elif iext_boost < 0.5:
        keywords.extend(["low intensity", "subthreshold", "minimal"])
    
    # 关键词匹配打分
    scored = []
    for entry in knowledge_base:
        text = (entry["title"] + " " + entry["question"]).lower()
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > 0:
            scored.append((score, entry))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]

def rag_optimize(n_iterations=8):
    history = []
    rag_log = []
    best_reward = float('-inf')
    best_params = None
    
    print("=== RAG-Augmented LLM Optimizer ===\n")
    
    # baseline
    r0, _, _ = run_robust_clinical(0.0, 9)
    print(f"Baseline: {r0:.4f}\n")
    
    for i in range(n_iterations):
        # 第0次用文献先验初始化
        if i == 0:
            params = {"iext_boost": 1.5, "site_index": 9}
            trend = "improving"
        else:
            # 判断趋势
            recent = [h["reward"] for h in history[-3:]]
            if len(recent) >= 2:
                if recent[-1] > recent[-2]:
                    trend = "improving"
                elif recent[-1] < recent[-2] - 0.002:
                    trend = "worsening"
                else:
                    trend = "plateau"
            else:
                trend = "improving"
            
            # RAG检索
            site_name = labels[history[-1]["site_index"]] if history else "rHC"
            last_boost = history[-1]["iext_boost"] if history else 1.5
            retrieved = retrieve_knowledge(site_name, last_boost, trend)
            
            rag_context = "\n".join([
                f"[PMID {r['pmid']}] {r['title'][:60]}...\n  Gap: {r['question'][:120]}"
                for r in retrieved
            ])
            
            # 构建dual-loop prompt
            history_str = json.dumps([{
                "iter": h["iteration"], "boost": h["iext_boost"],
                "site": h["site_name"], "reward": h["reward"]
            } for h in history], indent=2)
            
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=600,
                messages=[{"role": "user", "content": f"""You are optimizing brain stimulation parameters for epilepsy treatment.

[QUANTITATIVE FEEDBACK - TVB Simulation Results]
Baseline reward: {r0:.4f}
Optimization history:
{history_str}
Current trend: {trend}

[QUALITATIVE FEEDBACK - Retrieved Medical Knowledge from 136-paper literature review]
{rag_context}

[NETWORK TOPOLOGY CONTEXT]
rHC (index 9): degree=1, strength=2.0 — isolated node, no direct fiber connections to SOZ regions
rFEF (index 7): degree=26, strength=45.0 — true network hub
Top network hubs: rPFCORB, lPFCORB, rPFCVL (prefrontal regions)

[TASK]
Based on simulation results AND retrieved literature knowledge, propose next parameters.
Constraints: iext_boost 0.0-4.0, site_index 0-75 (integer), avoid boost>3.5 (safety).

Respond with JSON only:
{{"reasoning": "<cite specific literature insight and how it guides your choice>", "iext_boost": <float>, "site_index": <int>}}"""}]
            )
            
            text = response.content[0].text.strip()
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            parsed = json.loads(match.group())
            params = {"iext_boost": float(parsed["iext_boost"]),
                      "site_index": int(parsed["site_index"])}
            reasoning = parsed.get("reasoning", "")
            
            rag_log.append({
                "iteration": i,
                "trend": trend,
                "retrieved_papers": [{"pmid": r["pmid"], "title": r["title"][:80], 
                                       "gap": r["question"][:120]} for r in retrieved],
                "reasoning": reasoning,
                "params": params
            })
        
        # 跑simulation
        worst, mean, _ = run_robust_clinical(
            params["iext_boost"], params["site_index"])
        
        if worst > best_reward:
            best_reward = worst
            best_params = params.copy()
        
        site_name = labels[params["site_index"]]
        history.append({
            "iteration": i,
            "iext_boost": round(params["iext_boost"], 2),
            "site_index": params["site_index"],
            "site_name": site_name,
            "reward": round(worst, 4)
        })
        
        print(f"Iter {i}: boost={params['iext_boost']:.2f} site={params['site_index']}({site_name}) "
              f"| reward={worst:.4f} best={best_reward:.4f}")
    
    improvement = (best_reward - r0) / abs(r0) * 100
    print(f"\nBest reward: {best_reward:.4f} ({improvement:.1f}% improvement)")
    
    result = {
        "baseline": round(r0, 4),
        "best_reward": round(best_reward, 4),
        "improvement_pct": round(improvement, 1),
        "best_params": best_params,
        "history": history,
        "rag_log": rag_log
    }
    with open("rag_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to rag_results.json")
    return result

if __name__ == "__main__":
    rag_optimize()
