"""Ad-hoc cross-check of generators, solver and isomorphism against independent references."""
import time, collections
import networkx as nx
from trace_vr.rng import Rng
from trace_vr import families, semantics as sem, reference as ref, iso

t0=time.time()
stats=collections.Counter(); twinstats=collections.Counter(); kinds=collections.Counter()
for fam in families.FAMILIES:
    for i in range(120):
        rng=Rng("t",fam,i)
        n=rng.randint(5,10) if fam in("layered_dag","tree_plus","sparse_random") else rng.randint(11,16)
        g=families.generate(fam,n,rng.child("g"))
        ok,why=families.family_invariants_ok(g); assert ok,(fam,why)
        G=nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(g.edges)
        for s in range(n):
            desc=set()
            for v in G.successors(s):
                desc.add(v); desc|=nx.descendants(G,v)
            for t in range(n):
                assert g.reaches(s,t)==(t in desc),(fam,i,s,t)
        nodes=[str(k) for k in range(n)]; edges=[(str(u),str(v)) for u,v in g.edges]
        L=lambda ps:[(str(a),str(b)) for a,b in ps]
        for cat in sem.REPAIR_CATEGORIES:
            spec=None
            for a in range(40):
                spec=sem.propose_repair(g,rng.child("rep",cat,a),cat)
                if spec: break
            stats[(fam,cat,spec is not None)]+=1
            if not spec: continue
            st,sols,mn=ref.repair(nodes,edges,L(spec["must_reach"]),L(spec["must_not"]),L(spec["additions"]))
            sol=spec["solution"]
            assert st==sol.status and mn==sol.min_edits
            assert sorted(sols)==sorted([[(op,str(u),str(v)) for op,u,v in s] for s in sol.solutions]),(sols,sol.solutions)
            tw=sem.repair_twin_mods(g,spec)
            twinstats[(cat,len(tw)>0)]+=1
            for mod,s2 in tw[:3]:
                kinds[(cat,sem.repair_category(s2))]+=1
        for ans in (True,False):
            r=sem.propose_reachability(g,rng.child("reach",ans),ans)
            stats[("reach",ans,bool(r and r["twin_mods"]))]+=1
            if r:
                assert g.reaches(r["src"],r["dst"])==ans
                for mod in r["twin_mods"][:5]:
                    assert sem.apply_mod(g,mod).reaches(r["src"],r["dst"])!=ans
            c=sem.propose_critical_edge(g,rng.child("crit",ans),ans)
            stats[("crit",ans,bool(c and c["twin_mods"]))]+=1
            if c:
                assert sem.critical_answer(g,c["src"],c["dst"],c["edge"])==ans
                assert ref.critical(nodes,edges,str(c["src"]),str(c["dst"]),(str(c["edge"][0]),str(c["edge"][1])))==ans
                for mod in c["twin_mods"][:5]:
                    assert sem.critical_answer(sem.apply_mod(g,mod),c["src"],c["dst"],c["edge"])!=ans
        perm=rng.shuffled(range(n)); g2=g.relabeled(perm)
        assert iso.fingerprint(g)==iso.fingerprint(g2) and iso.is_isomorphic(g,g2)
        mods=sem.single_edge_mods(g); m=rng.choice(mods); g3=sem.apply_mod(g,m).relabeled(perm)
        G3=nx.DiGraph(); G3.add_nodes_from(range(n)); G3.add_edges_from(g3.edges)
        assert iso.is_isomorphic(g,g3)==nx.is_isomorphic(G,G3)
print("time %.1fs"%(time.time()-t0))
for k in sorted(stats,key=str): print(k,stats[k])
print("twin availability",dict(twinstats))
print("twin category by base",sorted(kinds.items()))
