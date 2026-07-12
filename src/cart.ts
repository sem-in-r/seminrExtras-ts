/**
 * Internal CART engine with exact rpart (anova method) parity.
 *
 * A port of the subset of rpart 4.1.24 that COA consumes (py plan F3/F15):
 * regression trees grown with `rpart(y ~ ., minsplit = 2, cp = 0)`,
 * reproducing rpart's `frame` (power-of-2 node numbering, depth-first order,
 * per-node complexity), `splits` (primary + competitor + surrogate blocks) and
 * `where` (frame-row leaf assignment) exactly. The algorithm follows the rpart
 * C sources (anova.c, bsplit.c, insert_split.c, partition.c, surrogate.c,
 * choose_surg.c) via the py port's cart.py; every rule below cites that origin.
 * No missing values and unit weights are assumed — COA's predictive-deviance
 * data satisfies both.
 *
 * NOT exported from the package barrel (internal engine, R `:::` analog);
 * COA imports `rpartAnova` directly.
 *
 * Key parity rules (py plan F15):
 *
 * - Split goodness per candidate boundary: `left_sum^2/left_n + right_sum^2/right_n`
 *   of grand-mean-centered y, scanned in ascending-x order with strict `>`
 *   (first best wins within a variable); split point = midpoint of the adjacent
 *   x values; direction from centered SUMS (not means): left child = low-x side
 *   iff `left_sum < right_sum` (`ncat = -1`), else the low-x side goes right
 *   (`ncat = +1`).
 * - Variables are tried in column order; a variable's best split is accepted
 *   iff its scaled improvement exceeds `1e-10 x` the largest improvement seen
 *   so far ANYWHERE in the tree build (rpart's global `iscale`); accepted
 *   splits go into a descending-improve list capped at `maxcompete + 1` where
 *   ties keep insertion (= variable) order. Head = primary, rest = competitors.
 * - Surrogates: per other variable, the split that best agrees with the
 *   primary's left/right assignment, requiring at least 2 observations on each
 *   surrogate side and agreement strictly above the majority baseline; kept iff
 *   adjusted agreement `> 1e-10`; ordered by agreement (descending, ties =
 *   variable order), capped at `maxsurrogate`. In `splits` rows, `improve`
 *   holds the agreement fraction and `count` stays 0 on complete data.
 * - Node complexity: computed bottom-up as mean risk reduction per split with
 *   rpart's collapse rule, capped by the inherited complexity (make_cp_list
 *   post-pass: no node exceeds its parent), divided by the root deviance at
 *   the frame level; leaves show `cp`. A zero-deviance root scales 0/0 = NaN,
 *   exactly as rpart reports it.
 */

import { seqSum } from "./helpers.ts";
import { gFormat } from "./records.ts";

const LEFT = -1;
const RIGHT = 1;

/**
 * Node coordinates and edges for an rpart `plot(uniform = TRUE)` diagram.
 *
 * Coordinates run in frame (preorder) order. Leaves are spread at `x = 1..`
 * in preorder; each internal node sits at the mean `x` of its two children.
 * `y` decreases with depth so the root is at the top. `edges` holds one
 * square-shouldered polyline per parent->child link: `[parent, elbow, child]`
 * with the elbow at the child's `x` and the parent's `y` (rpart branch = 1).
 */
export interface CartLayout {
  readonly nodeIds: readonly number[];
  readonly x: readonly number[];
  readonly y: readonly number[];
  readonly isLeaf: readonly boolean[];
  readonly edges: readonly (readonly [
    readonly [number, number],
    readonly [number, number],
    readonly [number, number],
  ])[];
}

/**
 * An rpart-parity regression tree (mirrors `tree$frame`/`$splits`/`$where`).
 *
 * Frame vectors run in rpart's depth-first, left-child-first order; the
 * `splits` vectors hold one block per internal frame row (primary, then
 * `ncompete` competitors, then `nsurrogate` surrogates). `where` gives each
 * observation's leaf as a 1-based frame row index (NOT a node id).
 */
export interface CartTree {
  readonly kind: "cart_tree";
  // frame columns (row names = nodeIds)
  readonly nodeIds: readonly number[];
  readonly var: readonly string[];
  readonly n: readonly number[];
  readonly wt: readonly number[];
  readonly dev: readonly number[];
  readonly yval: readonly number[];
  readonly complexity: readonly number[];
  readonly ncompete: readonly number[];
  readonly nsurrogate: readonly number[];
  // splits columns (row names = splitVar)
  readonly splitVar: readonly string[];
  readonly splitCount: readonly number[];
  readonly splitNcat: readonly number[];
  readonly splitImprove: readonly number[];
  readonly splitIndex: readonly number[];
  readonly splitAdj: readonly number[];
  /** Per-observation leaf assignment (1-based frame row index). */
  readonly where: readonly number[];
  /**
   * 1-based start offset of each frame row's split block (R `tree_split_index`).
   *
   * Leaves occupy zero rows; the offset sequence is the cumulative sum COA
   * computes as `cumsum(c(1, ncompete + nsurrogate + !is_leaf))`.
   */
  splitRowOffsets(): number[];
  /**
   * Node positions and square-shouldered edges (rpart `plot(uniform=TRUE)`).
   * Layout math lives here so the plot layer only draws.
   */
  plotLayout(): CartLayout;
  /**
   * Left-branch split rule per internal node id (`text.rpart` label format):
   * `ncat < 0` means the left child takes `x < index` (label `"<var>< <value>"`),
   * else the left child takes `x >= index` (label `"<var>>=<value>"`).
   */
  splitRuleLabels(): Record<number, string>;
  /** Per-node-id value label (rounded `yval`), leaves adding `"n=<n>"`. */
  nodeLabels(useN?: boolean): Record<number, string>;
  toString(): string;
}


/** One splits-table row candidate (primary/competitor/surrogate alike). */
interface SplitRec {
  var: number;
  /** Scaled improvement, or agreement fraction for surrogates. */
  improve: number;
  spoint: number;
  /** ncat column: -1 => x < spoint goes left, +1 => goes right. */
  csplit0: number;
  count: number;
  adj: number;
}

interface Node {
  nodeId: number;
  idx: number[]; // observation indices in this node
  risk: number;
  yval: number;
  complexity: number;
  primaries: SplitRec[];
  surrogates: SplitRec[];
  left: Node | null;
  right: Node | null;
}

function makeNode(nodeId: number, idx: number[]): Node {
  return {
    nodeId,
    idx,
    risk: 0,
    yval: 0,
    complexity: 0,
    primaries: [],
    surrogates: [],
    left: null,
    right: null,
  };
}

/** Mean and SS with rpart's sequential accumulation (anova.c `anovass`). */
function nodeStats(y: readonly number[]): [number, number] {
  const mean = seqSum(y) / y.length;
  const ss = seqSum(y.map((v) => (v - mean) * (v - mean)));
  return [mean, ss];
}

/**
 * Best anova split on one sorted variable (anova.c `anova`, continuous branch).
 *
 * Returns `[scaledImprove, splitPoint, direction]`; improve 0 means no valid
 * position. `xs` must be ascending with `ys` aligned.
 */
function anovaScan(
  xs: readonly number[],
  ys: readonly number[],
  myrisk: number,
  edge: number,
): [number, number, number] {
  const n = xs.length;
  const grand = seqSum(ys) / n;
  // left_sum after obs 0..i inclusive, sequential accumulation like np.cumsum
  let best = -Infinity;
  let where = -1;
  let acc = 0;
  const leftSums: number[] = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    acc += ys[i]! - grand;
    leftSums[i] = acc;
    const leftWt = i + 1;
    const rightWt = n - 1 - i;
    // rpart evaluates at i where the next x differs, left_n >= edge, right_n >= edge
    if (xs[i + 1] !== xs[i] && leftWt >= edge && rightWt >= edge) {
      const sq = acc * acc; // right_sum = -left_sum bitwise, so squares coincide
      const goodness = sq / leftWt + sq / rightWt;
      if (goodness > best) {
        best = goodness;
        where = i;
      }
    }
  }
  if (where < 0 || best <= 0) return [0, NaN, 0];
  const direction = leftSums[where]! < 0 ? LEFT : RIGHT;
  const split = (xs[where]! + xs[where + 1]!) / 2;
  return [best / myrisk, split, direction];
}

/** insert_split.c: descending-improve insertion, strict `>` (ties keep order). */
function insertSorted(splits: SplitRec[], rec: SplitRec, cap: number): void {
  let pos = splits.length;
  for (let k = 0; k < splits.length; k++) {
    if (rec.improve > splits[k]!.improve) {
      pos = k;
      break;
    }
  }
  if (pos >= cap) return;
  splits.splice(pos, 0, rec);
  splits.length = Math.min(splits.length, cap);
}

/** Stable ascending sort order of `xv` (indices). */
function stableOrder(xv: readonly number[]): number[] {
  return Array.from({ length: xv.length }, (_, i) => i).sort((a, b) => xv[a]! - xv[b]!);
}

/**
 * Best surrogate split on one variable (choose_surg.c, continuous branch).
 *
 * `tempy` holds the primary's assignment (-1 left / +1 right) per node
 * observation; returns `[agreement, splitPoint, direction, adj]` or null when
 * no split with >= 2 observations per side beats the majority.
 */
function chooseSurg(
  xv: readonly number[],
  tempy: readonly number[],
  lcount: number,
  rcount: number,
): [number, number, number, number] | null {
  const order = stableOrder(xv);
  const xs = order.map((i) => xv[i]!);
  const ty = order.map((i) => tempy[i]!);
  let ll = 0;
  let rl = 0;
  for (const t of ty) {
    if (t === LEFT) ll++;
    else if (t === RIGHT) rl++;
  }
  let llwt = ll;
  let rlwt = rl;
  let agree = Math.max(llwt, rlwt);
  let lr = 0;
  let rr = 0;
  let lrwt = 0;
  let rrwt = 0;
  let success = false;
  let csplit0 = LEFT;
  let split = xs[0]!;
  let lastx = xs[0]!;
  let i = 0;
  while (ll + rl >= 2) {
    const xj = xs[i]!;
    if (lr + rr >= 2 && xj !== lastx) {
      if (llwt + rrwt > agree) {
        success = true;
        agree = llwt + rrwt;
        csplit0 = RIGHT; // x < split goes to the right
        split = (xj + lastx) / 2;
      } else if (lrwt + rlwt > agree) {
        success = true;
        agree = lrwt + rlwt;
        csplit0 = LEFT;
        split = (xj + lastx) / 2;
      }
    }
    if (ty[i] === LEFT) {
      ll -= 1;
      lr += 1;
      llwt -= 1;
      lrwt += 1;
    } else if (ty[i] === RIGHT) {
      rl -= 1;
      rr += 1;
      rlwt -= 1;
      rrwt += 1;
    }
    lastx = xj;
    i += 1;
  }
  if (!success) return null;
  // surrogatestyle = 0: totals over the full node (choose_surg.c sur_agree == 0)
  const totalWt = lcount + rcount;
  const majority = Math.max(lcount, rcount) / totalWt;
  const agreement = agree / totalWt;
  const adj = (agreement - majority) / (1 - majority);
  if (adj <= 1e-10) return null;
  return [agreement, split, csplit0, adj];
}

interface BuilderOptions {
  minsplit: number;
  minbucket: number;
  cp: number;
  maxcompete: number;
  maxsurrogate: number;
  maxdepth: number;
}

class Builder {
  readonly root: Node;
  readonly rootRisk: number;
  private readonly maxnode: number;
  private readonly alpha: number;
  private iscale = 0; // rpart.c: reset per tree; bsplit.c running max improve

  constructor(
    private readonly x: readonly (readonly number[])[],
    private readonly y: readonly number[],
    private readonly nVar: number,
    private readonly opts: BuilderOptions,
  ) {
    this.maxnode = 2 ** opts.maxdepth - 1; // rpart.c: nodenum > maxnode => forced leaf
    const [, rootRisk] = nodeStats(y);
    this.alpha = opts.cp * rootRisk; // rp.alpha
    this.root = makeNode(1, Array.from({ length: y.length }, (_, i) => i));
    this.rootRisk = rootRisk;
    // partition(): root inherits complexity = its own risk (rpart.c line 212)
    this.partition(this.root, rootRisk);
    // make_cp_list.c post-pass: a node collapses no later than its parent,
    // so complexity = min(own, parent's), floored at alpha for the descent
    this.capComplexity(this.root, this.root.complexity);
  }

  private capComplexity(node: Node, parent: number): void {
    if (node.complexity > parent) node.complexity = parent;
    const meCp = Math.max(node.complexity, this.alpha);
    if (node.left !== null && node.right !== null) {
      this.capComplexity(node.left, meCp);
      this.capComplexity(node.right, meCp);
    }
  }

  // -- bsplit.c ---------------------------------------------------------------

  private bsplit(node: Node): SplitRec[] {
    const primaries: SplitRec[] = [];
    const yv = node.idx.map((i) => this.y[i]!);
    const k = node.idx.length;
    for (let v = 0; v < this.nVar; v++) {
      const xv = node.idx.map((i) => this.x[i]![v]!);
      const order = stableOrder(xv);
      const xs = order.map((i) => xv[i]!);
      if (xs[0] === xs[xs.length - 1]) continue; // no place to split
      const ys = order.map((i) => yv[i]!);
      const [improve, split, direction] = anovaScan(xs, ys, node.risk, this.opts.minbucket);
      if (improve > this.iscale) this.iscale = improve;
      if (improve > this.iscale * 1e-10) {
        insertSorted(
          primaries,
          { var: v, improve, spoint: split, csplit0: direction, count: k, adj: 0 },
          this.opts.maxcompete + 1,
        );
      }
    }
    return primaries;
  }

  // -- surrogate.c --------------------------------------------------------------

  private surrogate(node: Node): SplitRec[] {
    const primary = node.primaries[0]!;
    const xp = node.idx.map((i) => this.x[i]![primary.var]!);
    // tempy: the primary's left(-1)/right(+1) assignment per observation
    const lowLeft = primary.csplit0 === LEFT;
    const tempy = xp.map((v) => ((v < primary.spoint) === lowLeft ? LEFT : RIGHT));
    let lcount = 0;
    let rcount = 0;
    for (const t of tempy) {
      if (t === LEFT) lcount++;
      else rcount++;
    }
    const surrogates: SplitRec[] = [];
    for (let v = 0; v < this.nVar; v++) {
      if (v === primary.var) continue;
      const found = chooseSurg(
        node.idx.map((i) => this.x[i]![v]!),
        tempy,
        lcount,
        rcount,
      );
      if (found === null) continue;
      const [agreement, split, direction, adj] = found;
      insertSorted(
        surrogates,
        { var: v, improve: agreement, spoint: split, csplit0: direction, count: 0, adj },
        this.opts.maxsurrogate,
      );
    }
    return surrogates;
  }

  // -- partition.c --------------------------------------------------------------

  /** Grow `me` recursively; returns [subtree risk, subtree split count]. */
  private partition(me: Node, inherited: number): [number, number] {
    const [yval, risk] = nodeStats(me.idx.map((i) => this.y[i]!));
    me.yval = yval;
    me.risk = risk;
    let tempcp = me.nodeId === 1 ? me.risk : Math.min(me.risk, inherited);

    if (me.idx.length < this.opts.minsplit || tempcp <= this.alpha || me.nodeId > this.maxnode) {
      me.complexity = this.alpha;
      return [me.risk, 0];
    }

    me.primaries = this.bsplit(me);
    if (me.primaries.length === 0) {
      me.complexity = this.alpha;
      return [me.risk, 0];
    }
    if (this.opts.maxsurrogate > 0) me.surrogates = this.surrogate(me);

    const primary = me.primaries[0]!;
    const xp = me.idx.map((i) => this.x[i]![primary.var]!);
    const lowLeft = primary.csplit0 === LEFT;
    const leftIdx: number[] = [];
    const rightIdx: number[] = [];
    me.idx.forEach((obs, k) => {
      if ((xp[k]! < primary.spoint) === lowLeft) leftIdx.push(obs);
      else rightIdx.push(obs);
    });
    me.left = makeNode(2 * me.nodeId, leftIdx);
    me.right = makeNode(2 * me.nodeId + 1, rightIdx);

    let [leftRisk, leftSplit] = this.partition(me.left, tempcp - this.alpha);

    let tempcpR = Math.max((me.risk - leftRisk) / (leftSplit + 1), me.risk - me.left.risk);
    tempcpR = Math.min(tempcpR, inherited);
    let [rightRisk, rightSplit] = this.partition(me.right, tempcpR - this.alpha);

    // collapse rule: children with smaller complexity than my tentative cp
    // count as unsplit for my cp computation
    tempcp = (me.risk - (leftRisk + rightRisk)) / (leftSplit + rightSplit + 1);
    if (me.right.complexity > me.left.complexity) {
      if (tempcp > me.left.complexity) {
        leftRisk = me.left.risk;
        leftSplit = 0;
        tempcp = (me.risk - (leftRisk + rightRisk)) / (leftSplit + rightSplit + 1);
        if (tempcp > me.right.complexity) {
          rightRisk = me.right.risk;
          rightSplit = 0;
        }
      }
    } else if (tempcp > me.right.complexity) {
      rightRisk = me.right.risk;
      rightSplit = 0;
      tempcp = (me.risk - (leftRisk + rightRisk)) / (leftSplit + rightSplit + 1);
      if (tempcp > me.left.complexity) {
        leftRisk = me.left.risk;
        leftSplit = 0;
      }
    }
    me.complexity = (me.risk - (leftRisk + rightRisk)) / (leftSplit + rightSplit + 1);

    if (me.complexity <= this.alpha) {
      // "all was in vain" — unreachable at cp = 0 (an accepted split has
      // strictly positive deviance reduction) but kept for faithfulness
      me.left = null;
      me.right = null;
      me.primaries = [];
      me.surrogates = [];
      return [me.risk, 0];
    }
    return [leftRisk + rightRisk, leftSplit + rightSplit + 1];
  }
}

/** Options of {@link rpartAnova}, mirroring `rpart.control` fields it honors. */
export interface RpartAnovaOptions {
  minsplit?: number;
  cp?: number;
  /** Defaults to `round(minsplit / 3)` like `rpart.control`. */
  minbucket?: number;
  maxcompete?: number;
  maxsurrogate?: number;
  maxdepth?: number;
}

/**
 * Grow an rpart-parity anova regression tree (`rpart(y ~ ., ...)`).
 *
 * `x` is the n x p predictor matrix with columns named by `varNames`
 * (model-frame order — ties between equally good variables resolve to the
 * earlier column, as in rpart). Complete data and unit weights are assumed.
 */
export function rpartAnova(
  x: readonly (readonly number[])[],
  y: readonly number[],
  varNames: readonly string[],
  options: RpartAnovaOptions = {},
): CartTree {
  if (x.length !== y.length || y.length === 0) {
    throw new Error("x must be an n x p matrix aligned with the n-vector y.");
  }
  const nVar = x[0]!.length;
  if (nVar !== varNames.length) {
    throw new Error("varNames must name every column of x.");
  }
  if (x.some((row) => row.some(Number.isNaN)) || y.some(Number.isNaN)) {
    throw new Error("The internal CART engine requires complete data (no NaN).");
  }
  const minsplit = options.minsplit ?? 2;
  // R round() is half-even, matching Math.round only away from .5 ties; the
  // only tie in practice is minsplit=2 -> round(2/3)=1, which agrees.
  const minbucket = Math.max(options.minbucket ?? Math.round(minsplit / 3), 1);

  const builder = new Builder(x, y, nVar, {
    minsplit,
    minbucket,
    cp: options.cp ?? 0,
    maxcompete: options.maxcompete ?? 4,
    maxsurrogate: options.maxsurrogate ?? 5,
    maxdepth: options.maxdepth ?? 30,
  });

  // depth-first, left-first frame assembly
  const nodeIds: number[] = [];
  const varCol: string[] = [];
  const nCol: number[] = [];
  const wt: number[] = [];
  const dev: number[] = [];
  const yval: number[] = [];
  const complexity: number[] = [];
  const ncompete: number[] = [];
  const nsurrogate: number[] = [];
  const splitVar: string[] = [];
  const splitCount: number[] = [];
  const splitNcat: number[] = [];
  const splitImprove: number[] = [];
  const splitIndex: number[] = [];
  const splitAdj: number[] = [];
  const where: number[] = new Array(y.length).fill(0);
  const rootRisk = builder.rootRisk;

  const emit = (node: Node): void => {
    const row = nodeIds.length;
    nodeIds.push(node.nodeId);
    nCol.push(node.idx.length);
    wt.push(node.idx.length);
    dev.push(node.risk);
    yval.push(node.yval);
    // R scales by the root deviance unconditionally: 0/0 = NaN on a
    // zero-deviance root, exactly as rpart reports it
    complexity.push(rootRisk !== 0 ? node.complexity / rootRisk : NaN);
    if (node.left === null) {
      varCol.push("<leaf>");
      ncompete.push(0);
      nsurrogate.push(0);
      for (const obs of node.idx) where[obs] = row + 1;
      return;
    }
    varCol.push(varNames[node.primaries[0]!.var]!);
    ncompete.push(node.primaries.length - 1);
    nsurrogate.push(node.surrogates.length);
    for (const rec of [...node.primaries, ...node.surrogates]) {
      splitVar.push(varNames[rec.var]!);
      splitCount.push(rec.count);
      splitNcat.push(rec.csplit0);
      splitImprove.push(rec.improve);
      splitIndex.push(rec.spoint);
      splitAdj.push(rec.adj);
    }
    emit(node.left);
    emit(node.right!);
  };

  emit(builder.root);

  return Object.freeze({
    kind: "cart_tree" as const,
    nodeIds,
    var: varCol,
    n: nCol,
    wt,
    dev,
    yval,
    complexity,
    ncompete,
    nsurrogate,
    splitVar,
    splitCount,
    splitNcat,
    splitImprove,
    splitIndex,
    splitAdj,
    where,
    splitRowOffsets(): number[] {
      const offsets: number[] = [];
      let pos = 1;
      for (let i = 0; i < nodeIds.length; i++) {
        offsets.push(pos);
        if (varCol[i] !== "<leaf>") pos += 1 + ncompete[i]! + nsurrogate[i]!;
      }
      return offsets;
    },
    plotLayout(): CartLayout {
      const frameRow = new Map<number, number>();
      nodeIds.forEach((nid, i) => frameRow.set(nid, i));
      const leaf = (nid: number): boolean => varCol[frameRow.get(nid)!] === "<leaf>";
      const depth = (nid: number): number => Math.floor(Math.log2(nid));
      const maxDepth = Math.max(...nodeIds.map(depth));
      const xpos = new Map<number, number>();
      let counter = 0;
      const assign = (nid: number): number => {
        let x: number;
        if (leaf(nid)) {
          counter += 1;
          x = counter;
        } else {
          x = (assign(2 * nid) + assign(2 * nid + 1)) / 2;
        }
        xpos.set(nid, x);
        return x;
      };
      assign(1);
      const yOf = (nid: number): number => maxDepth - depth(nid);
      const edges: [
        readonly [number, number],
        readonly [number, number],
        readonly [number, number],
      ][] = [];
      for (const nid of nodeIds) {
        if (leaf(nid)) continue;
        for (const child of [2 * nid, 2 * nid + 1]) {
          edges.push([
            [xpos.get(nid)!, yOf(nid)],
            [xpos.get(child)!, yOf(nid)],
            [xpos.get(child)!, yOf(child)],
          ]);
        }
      }
      return {
        nodeIds: [...nodeIds],
        x: nodeIds.map((nid) => xpos.get(nid)!),
        y: nodeIds.map(yOf),
        isLeaf: nodeIds.map(leaf),
        edges,
      };
    },
    splitRuleLabels(): Record<number, string> {
      const offsets = this.splitRowOffsets();
      const labels: Record<number, string> = {};
      for (let i = 0; i < nodeIds.length; i++) {
        if (varCol[i] === "<leaf>") continue;
        const row = offsets[i]! - 1;
        const variable = splitVar[row]!;
        const value = gFormat(splitIndex[row]!);
        labels[nodeIds[i]!] =
          splitNcat[row]! < 0 ? `${variable}< ${value}` : `${variable}>=${value}`;
      }
      return labels;
    },
    nodeLabels(useN = true): Record<number, string> {
      const labels: Record<number, string> = {};
      for (let i = 0; i < nodeIds.length; i++) {
        // display-only: JS toFixed(4) vs python round(x, 4) may differ on ties
        let label = gFormat(Number(yval[i]!.toFixed(4)));
        if (useN && varCol[i] === "<leaf>") label += `\nn=${nCol[i]}`;
        labels[nodeIds[i]!] = label;
      }
      return labels;
    },
    toString(): string {
      const nLeaves = varCol.filter((v) => v === "<leaf>").length;
      return (
        `CartTree: ${nodeIds.length} nodes (${nLeaves} leaves), ` +
        `${splitVar.length} split rows, n = ${where.length}`
      );
    },
  });
}
