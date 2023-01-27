# Competitive-Programming-Template

## Default code
```c++
#pragma GCC optimize("O3")
#include <bits/stdc++.h>
using namespace std;
#define FAST_IO  ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
#define show(x)  cout << ">> " << #x << " = " << x << '\n'
#define endl     '\n'
using ll = long long;
using ull = unsigned long long;
using ld = long double;
const double EPS = 1e-10;  // epsilon
const double PI = acos(-1);
const int MOD = 998244353;
const int BASE = 31;
const int INF = 0x3f3f3f3f;
const ll LL_INF = 0x3f3f3f3f3f3f3f3f;
const int dx[] = {0, 0, -1, 1};
const int dy[] = {-1, 1, 0, 0};
const int N = 1e5 + 5, M = 1e6 + 5;

void myMain()
{
    int n;
    cin >> n;
    vector<int> a(n);
    for (auto& item : a)
        cin >> item;
    
}

int main()
{
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    FAST_IO
    int t = 1;
    cin >> t;
    while (t--)
        myMain();
    return 0;
}
```
## DSU
```c++
class DSU {
private:
    vector<int> parent, sz;
 
public:
    DSU(int n)
    {
        parent = vector<int> (n + 1);
        sz = vector<int> (n + 1);
        for (int i = 1; i <= n; i++)
            makeSet(i);
    }
 
    void makeSet(int v)
    {
        parent[v] = v;
        sz[v] = 1;
    }
 
    bool sameSet(int a, int b)
    {
        return findParent(a) == findParent(b);
    }
 
    int findParent(int v)
    {
        if (parent[v] == v)
            return v;
        return parent[v] = findParent(parent[v]);
    }
 
    void unionSets(int a, int b)
    {
        a = findParent(a);
        b = findParent(b);
        if (sz[a] < sz[b])
            swap(a, b);
        if (a != b)
        {
            parent[b] = parent[a];
            sz[a] += sz[b];
        }
    }
 
    int getSize(int u)
    {
        return sz[u];
    }
};
```
## Segment Tree
```c++
const int IDN = 0;
ll segTree[4 * N], arr[N];
int oldn, n, q;
 
ll combine(ll a, ll b)
{
    return a + b;
}
 
void build(int& n)
{
    oldn = n;
    n = 1 << (__lg(n) + 1);
 
    for (int i = 0; i < oldn; i++)
        segTree[i + n] = arr[i];
 
    for (int i = n - 1; i; i--)
        segTree[i] = combine(segTree[i << 1], segTree[(i << 1) | 1]);
}
 
void update(int ql, int qr, ll v, int k = 1, int sl = 0, int sr = n - 1)
{
    if (sl == sr && sl == ql)
    {
        segTree[k] = v;
        return ;
    }
    if (qr < sl || sr < ql)
        return;
 
    int mid = (sl + sr) / 2;
    update(ql, qr, v, k << 1, sl, mid);
    update(ql, qr, v, (k << 1) | 1, mid + 1, sr);
 
    segTree[k] = combine(segTree[k << 1], segTree[(k << 1) | 1]);
}
 
ll query(int ql, int qr, int k = 1, int sl = 0, int sr = n - 1)
{
    if (ql <= sl && qr >= sr)
        return segTree[k];
 
    if (qr < sl || sr < ql)
        return IDN;
 
    int mid = (sl + sr) / 2;
    return combine(query(ql, qr, k << 1, sl, mid), query(ql, qr, k << 1 | 1, mid + 1, sr));
}
```
## BIT
```c++
class BIT {
private:
    int n;
    vector<ll> tree;
 
public:
    BIT(int n)
    {
        this->n = n;
        tree = vector<ll> (n + 2, 0);
    }
 
    void add(int idx, ll diff)
    {
        for (int i = idx; i <= n; i += (i & -i))
            tree[i] += diff;
    }
 
    ll prefixSum(int idx)
    {
        ll sum = 0;
        for (int i = idx; i > 0; i -= (i & -i))
            sum += tree[i];
        return sum;
    }
 
    ll rangeSum(int l, int r)
    {
        return prefixSum(r) - prefixSum(l - 1);
    }
};
```
## Mo's Algorithm
```c++
struct Query {
    int left, right, queryIndex, blockIndex;
 
    Query() {}
 
    Query(int l, int r, int qi, int sqRoot) : left(l), right(r), queryIndex(qi), blockIndex(left / sqRoot) {}
 
    bool operator < (const Query& query) const
    {
        if (this->blockIndex == query.blockIndex)
            return this->right < query.right;
        return this->blockIndex < query.blockIndex;
    }
};
 
int n, q, sqRoot, res, a[N], ans[M], freq[N];
map<int, int> id;
Query queries[M];
 
void add(int i)
{
    res += (++freq[a[i]] == 1);
}
 
void remove(int i)
{
    res -= (--freq[a[i]] == 0);
}
 
void moAlgo()
{
    sort(queries, queries + q);
    int l = 1, r = 0;
    for (int i = 0; i < q; i++)
    {
        while (l > queries[i].left)
            add(--l);
        while (l < queries[i].left)
            remove(l++);
        while (r < queries[i].right)
            add(++r);
        while (r > queries[i].right)
            remove(r--);
        ans[queries[i].queryIndex] = res;
    }
}
 
void myMain()
{
    cin >> n >> q;
    sqRoot = sqrt(n) + 1;
    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
        id[a[i]];
    }
    int sz = 0;
    for (auto& it : id)
        it.second = sz++;
    for (int i = 0; i < n; i++)
        a[i] = id[a[i]];
    for (int i = 0; i < q; i++)
    {
        int l, r;
        cin >> l >> r;
        queries[i] = Query(--l, --r, i, sqRoot);
    }
    moAlgo();
    for (int i = 0; i < q; i++)
        cout << ans[i] << endl;
}
```
## Square Root Decomposition
```c++

```
## Trie
```c++
class Trie {
private:
    struct Node {
        bool isLeaf;
        Node* child[26];
    };
    Node* root;

public:
    Trie()
    {
        root = newNode();
    }

    Node* newNode()
    {
        Node* node = new Node();
        for (int i = 0; i < 26; i++)
            node->child[i] = nullptr;
        node->isLeaf = true;
        return node;
    }

    void insert(string& s)
    {
        Node* node = root;
        for (auto& c : s)
        {
            if (node->child[c - 'a'] == nullptr)
            {
                node->isLeaf = false;
                node->child[c - 'a'] = newNode();
            }
            node = node->child[c - 'a'];
        }
    }

    bool find(string& s)
    {
        Node* node = root;
        for (auto& c : s)
        {
            if (node->child[c - 'a'] == nullptr)
                return false;
            node = node->child[c - 'a'];
        }
        return node->isLeaf;
    }

    bool isPrefix(string& s)
    {
        Node* node = root;
        for (auto& c : s)
            node = node->child[c - 'a'];
        return !node->isLeaf;
    }
};
```
## Ordered Set
```c++
// policy-based data structures
#include <ext/pb_ds/assoc_container.hpp>  // Common file
#include <ext/pb_ds/tree_policy.hpp>      // Including tree_order_statistics_node_update
using namespace __gnu_pbds;
 
template <typename T>
using ordered_set =
        tree<T,      // the type of the data that we want to insert (KEY)
        null_type, // the mapped policy
        less_equal<T>,  // the basis for comparison of two functions.
        rb_tree_tag,   // type of tree used (Red black)
        tree_order_statistics_node_update>; // contains various operations for updating the node variants of a tree-based container
```
## Dijkstra
```c++
vector<ll> dijkstra(int src)
{
    vector<ll> dis(n + 1, INF);
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;
    pq.emplace(dis[src] = 0, src);
    while (!pq.empty())
    {
        auto [w, u] = pq.top();
        pq.pop();
        if (w > dis[u])
            continue;
        for (auto& p : adj[u])
            if (dis[p.first] > dis[u] + p.second)
                pq.emplace(dis[p.first] = dis[u] + p.second, p.first);
    }
    return dis;
}
```
## Floyd-Warshall
```c++
void floyd()
{
    for (int k = 1; k <= n; k++)
        for (int u = 1; u <= n; u++)
            for (int v = 1; v <= n; v++)
                dis[u][v] = min(dis[u][v], dis[u][k] + dis[k][v]);
}
```
## Bellman-Ford
```c++
pair<ll, bool> bellman(int src, int dest)
{
    vector<ll> dis(n + 1, LL_INF);
    dis[src] = 0;
    for (int i = 1; i < n; i++)
    {
        bool updated = false;
        for (auto& edge : edges)
        {
            if (dis[edge.from] == LL_INF)
                continue;
            if (dis[edge.to] > dis[edge.from] + edge.cost)
            {
                dis[edge.to] = dis[edge.from] + edge.cost;
                updated = true;
            }
        }
        if (!updated)
            break;
    }
    vector<bool> hasCycle(n + 1);
    for (int i = 1; i < n; i++)
    {
        for (auto& edge: edges)
        {
            if (dis[edge.from] == LL_INF)
                continue;
            if (dis[edge.to] > dis[edge.from] + edge.cost)
            {
                dis[edge.to] = dis[edge.from] + edge.cost;
                hasCycle[edge.to] = true;
            }
            if (hasCycle[edge.from])
                hasCycle[edge.to] = true;
        }
    }
    return {dis[dest], hasCycle[dest]};
}
```
## LCA
```c++
int depth[N], ancestor[N][M];
vector<int> adj[N];
 
void dfs(int u, int p)
{
    for (auto& v : adj[u])
    {
        if (v == p)
            continue;
        depth[v] = depth[u] + 1;
        ancestor[v][0] = u;
        for (int i = 1; i < M; i++)
            ancestor[v][i] = ancestor[ancestor[v][i - 1]][i - 1];
        dfs(v, u);
    }
}
 
int kth_ancestor(int u, int k)
{
    for (int i = 0; i < M; i++)
        if (k & (1 << i))
            u = ancestor[u][i];
    return u;
}
 
int LCA(int u, int v)
{
    if (depth[u] < depth[v])
        swap(u, v);
    u = kth_ancestor(u, depth[u] - depth[v]);
    if (u == v)
        return u;
    for (int i = M - 1; i >= 0; i--)
    {
        if (ancestor[u][i] != ancestor[v][i])
        {
            u = ancestor[u][i];
            v = ancestor[v][i];
        }
    }
    return ancestor[u][0];
}
```
## Flatten the tree
```c++
int n, tin[N], tout[N], timer;
vector<int> adj[N];
 
void flattenTree(int u, int p)
{
    tin[u] = ++timer;
    for (auto& v : adj[u])
    {
        if (v == p)
            continue;
        flattenTree(v, u);
    }
    tout[u] = ++timer;
}
```
## KMP
```c++
vector<int> KMP(string& s)
{
    int n = s.length(), border = 0;
    vector<int> prefix(n);
    for (int i = 1; i < n; i++)
    {
        while (border && s[i] != s[border])
            border = prefix[border - 1];
        prefix[i] = border += (s[i] == s[border]);
    }
    return prefix;
}
```
## Z Algorithm
```c++
vector<int> ZAlgo(string& s)
{
    int n = s.length();
    vector<int> Z(n);
    for (int i = 1, l = 0, r = 0; i < n; i++)
    {
        int k = i - l;
        if (i + Z[k] >= r)
        {
            l = i;
            r = max(r, i);
            while (r < n && s[r - l] == s[r])
                r++;
            Z[i] = r - l;
        }
        else
            Z[i] = Z[k];
    }
    return Z;
}
```
## Small-To-Large Merging
```c++
int n, color[N], ans[N];
set<int> st[N];
vector<int> adj[N];
 
void dfs(int u, int p)
{
    st[u].emplace(color[u]);
    for (auto& v : adj[u])
    {
        if (v == p)
            continue;
        dfs(v, u);
        if (st[u].size() < st[v].size())
            swap(st[u], st[v]);
        for (auto& x : st[v])
            st[u].emplace(x);
    }
    ans[u] = st[u].size();
}
```
## Heavy-Light Decomposition
```c++
int n, q, val[N];
int subTreeCnt[N], depth[N], ancestor[N][M];
int segTree[4 * N];
int head[N], pos[N], curPos;
vector<int> adj[N];
 
void dfs(int u, int p)
{
    subTreeCnt[u] = 1;
    for (auto& v : adj[u])
    {
        if (v == p)
            continue;
        depth[v] = depth[u] + 1;
        ancestor[v][0] = u;
        for (int i = 1; i < M; i++)
            ancestor[v][i] = ancestor[ancestor[v][i - 1]][i - 1];
        dfs(v, u);
        subTreeCnt[u] += subTreeCnt[v];
    }
}
 
int kth_ancestor(int u, int k)
{
    for (int i = 0; i < M; i++)
        if (k & (1 << i))
            u = ancestor[u][i];
    return u;
}
 
int LCA(int u, int v)
{
    if (depth[u] < depth[v])
        swap(u, v);
    u = kth_ancestor(u, depth[u] - depth[v]);
    if (u == v)
        return u;
    for (int i = M - 1; i >= 0; i--)
    {
        if (ancestor[u][i] != ancestor[v][i])
        {
            u = ancestor[u][i];
            v = ancestor[v][i];
        }
    }
    return ancestor[u][0];
}
 
int combine(int a, int b)
{
    return max(a, b);
}
 
void update(int ql, int qr, int v, int k = 1, int sl = 0, int sr = n - 1)
{
    if (sl == sr && sl == ql)
        return void(segTree[k] = v);
    if (qr < sl || sr < ql)
        return;
 
    int mid = (sl + sr) / 2;
    update(ql, qr, v, k << 1, sl, mid);
    update(ql, qr, v, (k << 1) | 1, mid + 1, sr);
    segTree[k] = combine(segTree[k << 1], segTree[(k << 1) | 1]);
}
 
int query(int ql, int qr, int k = 1, int sl = 0, int sr = n - 1)
{
    if (ql <= sl && qr >= sr)
        return segTree[k];
    if (qr < sl || sr < ql)
        return IDN;
 
    int mid = (sl + sr) / 2;
    return combine(query(ql, qr, k << 1, sl, mid), query(ql, qr, (k << 1) | 1, mid + 1, sr));
}
 
void decompose(int u, int p, int h)
{
    head[u] = h;
    pos[u] = curPos++;
    update(pos[u], pos[u], val[u]);
    int heavyChild = 0;
    for (auto& v : adj[u])
    {
        if (v == p)
            continue;
        if (subTreeCnt[v] > subTreeCnt[heavyChild])
            heavyChild = v;
    }
    if (!heavyChild)
        return;
    decompose(heavyChild, u, h);
    for (auto& v : adj[u])
    {
        if (v == p || v == heavyChild)
            continue;
        decompose(v, u, v);
    }
}
```
## Centroid Decomposition
```c++
int n, subTreeCnt[N];
vector<int> adj[N];
 
void dfs(int u, int p)
{
    subTreeCnt[u] = 1;
    for (auto& v : adj[u])
    {
        if (v == p)
            continue;
        dfs(v, u);
        subTreeCnt[u] += subTreeCnt[v];
    }
}
 
int getCentroid(int u, int p)
{
    for (auto& v : adj[u])
    {
        if (v == p)
            continue;
        if (subTreeCnt[v] * 2 > n)
            return getCentroid(v, u);
    }
    return u;
}
```
## fast power
```c++
ll fastPower(ll base, ll exp, ll mod)
{
    ll ans = 1;
    base %= mod;
    while (exp)
    {
        if (exp & 1)
            ans = (ans * base) % mod;
        exp >>= 1;
        base = (base * base) % mod;
    }
    return ans;
}
```
## nCr
```c++
ll fact[N];
 
ll fastPower(ll base, ll exp, ll mod)
{
    ll ans = 1;
    base %= mod;
    while (exp)
    {
        if (exp & 1)
            ans = (ans * base) % mod;
        exp >>= 1;
        base = (base * base) % mod;
    }
    return ans;
}
 
ll modInverse(ll x)
{
    return fastPower(x, MOD - 2, MOD);
}
 
ll nCr(ll n, ll r)
{
    if (n < r)
        return 0;
    if (r == 0)
        return 1;
    return (fact[n] * modInverse(fact[r]) % MOD * modInverse(fact[n - r]) % MOD) % MOD;
}

int main()
{
    fact[0] = 1;
    for (int i = 1; i < N; i++)
        fact[i] = (fact[i - 1] * i) % MOD;
}
```
## get divisors
```c++
vector<int> getDivisors(int x)
{
    vector<int> ans;
    for (int i = 1; i * i <= x; i++)
        if (x % i == 0)
        {
            ans.emplace_back(i);
            if (i * i != x)
                ans.emplace_back(x / i);
        }
    return ans;
}
```
## prime factors
```c++
set<ll> primeFactors(ll n)
{
    set<ll> ans;
    for (ll i = 2; i * i <= n; i++)
    {
        while (n % i == 0)
        {
            ans.emplace(i);
            n /= i;
        }
    }
    if (n != 1)
        ans.emplace(n);
    return ans;
}
```