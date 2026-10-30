
import json
from pathlib import Path

import numpy as np


class VectorStore:
    """
    一个简易的向量存储类。

    核心职责：
    1. 保存文本片段 chunks。
    2. 保存每个文本片段对应的向量 vectors。
    3. 根据用户查询向量，找出最相似的文本片段。
    4. 将向量和文本保存到本地文件。
    5. 从本地文件重新加载数据。

    注意：
    这只是一个简易的内存向量库，不是 FAISS、Milvus、
    Chroma 或其他专业向量数据库。
    """

    def __init__(self, dim: int = 1024):
        """
        初始化向量库对象。

        参数：
            dim:
                每个向量的维度，默认是 1024。

                例如，一个 Embedding 模型输出：
                [0.12, -0.35, 0.67, ..., 0.08]

                如果里面一共有 1024 个数字，
                那么这个向量的维度就是 1024。

        self:
            Python 中表示当前对象本身。
            self.dim 就是当前向量库对象的维度。
        """

        # 记录向量维度。
        # 后续添加向量、检索向量时，都需要检查维度是否一致。
        self.dim = dim

        # 创建一个空的二维 NumPy 数组，用于存储所有向量。
        #
        # shape = (0, dim)
        #
        # 假设 dim = 1024：
        # shape = (0, 1024)
        #
        # 含义：
        # 当前有 0 条向量，每条向量有 1024 个数字。
        #
        # 为什么是二维数组？
        # 因为多条向量需要按行排列：
        #
        # 第 0 行 -> 第 1 个文本片段的向量
        # 第 1 行 -> 第 2 个文本片段的向量
        # 第 2 行 -> 第 3 个文本片段的向量
        #
        # dtype=np.float32：
        # 每个数字使用 32 位浮点数存储。
        # 相比 NumPy 默认的 float64，通常可以节省一半向量内存。
        self.vectors = np.zeros((0, dim), dtype=np.float32)

        # 保存与向量逐条对应的文本片段。
        #
        # 类型是 list[dict]，例如：
        #
        # [
        #     {"chunk_id": "c1", "text": "苹果是一种水果"},
        #     {"chunk_id": "c2", "text": "香蕉是黄色的"}
        # ]
        #
        # 你原来的注解写的是 list[str]，
        # 但实际传入的是字典，因此这里修正为 list[dict]。
        #
        # 这里的字典还可以额外保存来源文件、页码、标题等元数据。
        self.chunks: list[dict] = []

    def __len__(self):
        """
        返回向量库当前存储的文本片段数量。

        有了这个方法，就可以直接使用：

            len(store)

        而不必每次都写：

            len(store.chunks)

        例如：
            store 中有 3 个文本片段
            len(store) 就返回 3。

        Python 的 len() 函数会自动调用对象的 __len__() 方法。
        """

        # 每条文本片段对应一条向量。
        # 因此文本片段的数量就是当前索引中存储的条目数量。
        return len(self.chunks)

    def add(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        """
        向向量库中批量添加向量和对应的文本片段。

        参数：
            vectors:
                二维 NumPy 数组，形状必须为 (N, dim)。

                N 表示本次添加多少条向量。
                dim 表示每条向量有多少个数字。

            chunks:
                长度必须为 N 的文本片段列表。

                chunks[i] 必须与 vectors[i] 对应。

        返回：
            None，表示直接修改当前 VectorStore 对象。
        """

        # 将传入的向量转换成 NumPy 数组，并统一为 float32。
        #
        # np.asarray()：
        # 如果输入已经是兼容的 NumPy 数组，可以直接使用；
        # 如果输入是列表等其他兼容对象，则转换成数组。
        #
        # dtype=np.float32：
        # 统一数据类型，避免不同精度的数据混合存储。
        vectors = np.asarray(vectors, dtype=np.float32)

        # 检查输入向量是否为二维数组。
        #
        # vectors.ndim：
        # 表示数组的维度数量。
        #
        # 正确的形状应该类似：
        # (3, 1024)
        #
        # 错误的形状示例：
        # (1024,)       一维数组，只有一条向量
        # (3, 512)      二维数组，但每条向量只有 512 维
        #
        # vectors.shape[1]：
        # 取得二维数组的第 2 个维度，也就是每条向量的长度。
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"向量维度应该为 (N, {self.dim})，"
                f"实际为 {vectors.shape}"
            )

        # 检查向量数量与文本片段数量是否相同。
        #
        # 假设：
        # vectors.shape = (3, 1024)
        # len(chunks) = 2
        #
        # 这意味着有 3 条向量，却只有 2 条文本。
        # 那么第 3 条向量就没有对应的文本，无法正确检索。
        #
        # 因此必须保证：
        # len(vectors) == len(chunks)
        if len(vectors) != len(chunks):
            raise ValueError(
                f"向量数量 {len(vectors)} "
                f"与文本块数量 {len(chunks)} 不匹配"
            )

        # 将新向量追加到已有向量的末尾。
        #
        # np.vstack()：
        # 沿着垂直方向，也就是按行拼接两个二维数组。
        #
        # 举例：
        #
        # 原来的 self.vectors：
        # [
        #     [1, 2],
        #     [3, 4]
        # ]
        #
        # 新的 vectors：
        # [
        #     [5, 6]
        # ]
        #
        # 拼接后：
        # [
        #     [1, 2],
        #     [3, 4],
        #     [5, 6]
        # ]
        #
        # 注意：
        # np.vstack() 会创建拼接后的新数组。
        # 如果频繁添加少量数据，这种方式效率可能不高。
        self.vectors = np.vstack([self.vectors, vectors])

        # 将新文本片段追加到原有列表末尾。
        #
        # extend() 和 append() 的区别：
        #
        # chunks.extend([a, b])
        # -> 把 a、b 两个元素分别加入列表。
        #
        # chunks.append([a, b])
        # -> 把整个 [a, b] 作为一个元素加入列表。
        #
        # 这里需要逐条追加，所以使用 extend()。
        #
        # 最重要的是保持对应关系：
        # self.vectors[i] <-> self.chunks[i]
        self.chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """
        根据查询向量，返回最相似的 top_k 条文本片段。

        参数：
            query_vector:
                用户问题经过 Embedding 模型生成的向量。
                必须包含 self.dim 个数字。

            top_k:
                希望返回多少条结果，默认返回 5 条。

        返回：
            列表，每个元素是一个二元组：

                (文本片段字典, 相似度分数)

            例如：
                [
                    ({"chunk_id": "c1", "text": "苹果是一种水果"}, 0.98),
                    ({"chunk_id": "c2", "text": "香蕉是黄色的"}, 0.42)
                ]

            分数越大，表示在当前相似度计算方式下越接近。
            但相似度分数不等于回答正确的概率。
        """

        # 如果向量库中没有任何文本，就没有可检索的内容。
        # 直接返回空列表，避免后续进行无意义的矩阵运算。
        if len(self.chunks) == 0:
            return []

        # 检查 top_k 是否有效。
        #
        # top_k = 0 或负数都没有合理的检索意义。
        if top_k <= 0:
            raise ValueError("top_k 必须是正整数")

        # 将查询向量转换成 float32，并调整为二维数组。
        #
        # 假设原始 query_vector 的形状是：
        # (1024,)
        #
        # reshape(1, -1) 后：
        # (1, 1024)
        #
        # reshape(1, -1) 中：
        # 1 表示只有 1 行，即只有一个查询向量。
        # -1 表示让 NumPy 自动推断这一行需要多少列。
        #
        # 为什么要变成二维？
        # 因为后面要执行矩阵乘法。
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        # 检查查询向量的维度是否与向量库一致。
        #
        # 例如：
        # 库里的向量维度是 1024，
        # 但查询向量只有 768 维。
        #
        # 两者无法进行预期的矩阵乘法，必须提前报错。
        if q.shape[1] != self.dim:
            raise ValueError(
                f"查询向量维度应该为 {self.dim}，"
                f"实际为 {q.shape[1]}"
            )

        # 将向量库中的所有向量与查询向量进行矩阵乘法。
        #
        # 假设：
        #
        # self.vectors.shape = (3, 1024)
        # q.shape            = (1, 1024)
        #
        # 对于矩阵乘法，需要将 q 转置：
        #
        # q.T.shape          = (1024, 1)
        #
        # 所以：
        #
        # (3, 1024) @ (1024, 1) = (3, 1)
        #
        # 这里 @ 是 Python 的矩阵乘法运算符。
        #
        # 每一行向量都会与查询向量计算一次点积。
        #
        # 如果向量都已经归一化到单位长度，
        # 点积就等于余弦相似度。
        #
        # 如果没有归一化，这里只是点积，不能直接称为余弦相似度。
        #
        # reshape(-1) 把结果从 (N, 1) 压成 (N,)。
        scores = (self.vectors @ q.T).reshape(-1)

        # 对相似度分数从大到小排序，并取前 top_k 个索引。
        #
        # np.argsort(scores)：
        # 默认返回按分数从小到大排列的索引。
        #
        # 例如：
        # scores = [0.2, 0.9, 0.5]
        #
        # np.argsort(scores)：
        # [0, 2, 1]
        #
        # [::-1]：
        # 把索引顺序反过来。
        #
        # [1, 2, 0]
        #
        # [:top_k]：
        # 只保留前 top_k 个索引。
        #
        # 注意：
        # 如果向量库只有 3 条记录，即使 top_k=5，
        # 最终也只会返回 3 条。
        order = np.argsort(scores)[::-1][:top_k]

        # 根据排序后的索引，取出对应的文本和分数。
        #
        # self.chunks[i]：
        # 第 i 条文本片段。
        #
        # scores[i]：
        # 这条文本对应的相似度分数。
        #
        # float(scores[i])：
        # 将 NumPy 浮点数转换为普通 Python float，
        # 让返回结果更方便打印、序列化和后续处理。
        #
        # 最终结果类似：
        # [
        #     ({"chunk_id": "c1", "text": "苹果是一种水果"}, 1.0),
        #     ({"chunk_id": "c2", "text": "香蕉是黄色的"}, 0.2)
        # ]
        return [
            (self.chunks[i], float(scores[i]))
            for i in order
        ]

    def save(self, path: str | Path) -> None:
        """
        将向量库保存到本地磁盘。

        参数：
            path:
                保存目录，例如：
                "/data/index/test_store"

        保存两个文件：
            vectors.npy -> 保存向量矩阵
            chunks.json -> 保存文本片段及其元数据

        为什么要保存两个文件？
        因为向量矩阵适合用 NumPy 的二进制格式存储，
        而文本片段和元数据适合用 JSON 保存。
        """

        # 将传入的路径转换为 Path 对象。
        #
        # Path 是 pathlib 模块提供的路径操作类。
        # 相比手动拼接字符串，它更适合跨平台处理文件路径。
        #
        # 你原来的代码写成：
        # directory = Path(directory)
        #
        # 这是一个 Bug，因为 directory 在这里没有定义。
        # 应当使用传入的参数 path。
        directory = Path(path)

        # 创建保存目录。
        #
        # parents=True：
        # 如果父级目录不存在，也一起创建。
        #
        # exist_ok=True：
        # 如果目录已经存在，不要因为目录存在而报错。
        #
        # 例如：
        # /data/index/test_store
        #
        # 如果 /data/index 不存在，也会尝试创建。
        directory.mkdir(parents=True, exist_ok=True)

        # 保存向量矩阵。
        #
        # np.save() 会将 NumPy 数组保存为 .npy 文件。
        #
        # 最终文件：
        # /data/index/test_store/vectors.npy
        #
        # 文件中保存的是数值矩阵，而不是文本内容。
        np.save(directory / "vectors.npy", self.vectors)

        # 保存文本片段。
        #
        # json.dumps()：
        # 将 Python 的列表、字典等数据结构转换为 JSON 字符串。
        #
        # ensure_ascii=False：
        # 不把中文强制转义为 \uXXXX 的形式，
        # 因此打开 JSON 文件时可以直接阅读中文。
        #
        # indent=2：
        # 使用两个空格缩进，让 JSON 文件更易读。
        #
        # encoding="utf-8"：
        # 使用 UTF-8 编码保存文本。
        #
        # write_text()：
        # 将字符串写入指定文件。
        (directory / "chunks.json").write_text(
            json.dumps(
                self.chunks,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    def load(self, directory: str | Path) -> None:
        """
        从本地磁盘加载已有的向量库。

        参数：
            directory:
                之前保存向量库的目录。

        加载过程：
            1. 读取 vectors.npy。
            2. 读取 chunks.json。
            3. 根据向量矩阵恢复维度。
            4. 检查数据是否对应。

        注意：
            这个方法会覆盖当前对象中已经存在的数据。
        """

        # 将目录路径转换为 Path 对象。
        directory = Path(directory)

        # 读取之前保存的向量矩阵。
        #
        # np.load()：
        # 读取 .npy 文件并恢复为 NumPy 数组。
        #
        # 原来保存的是 (3, 1024)，
        # 正常加载后仍然是 (3, 1024)。
        #
        # allow_pickle=False 是默认的安全选择：
        # 这里只需要读取数值数组，不需要反序列化任意 Python 对象。
        self.vectors = np.load(
            directory / "vectors.npy",
            allow_pickle=False
        ).astype(np.float32, copy=False)

        # 读取 JSON 文件中的文本片段。
        #
        # read_text()：
        # 将文件内容读取为字符串。
        #
        # json.loads()：
        # 将 JSON 字符串解析回 Python 列表和字典。
        #
        # 加载之后，原来的文本片段结构会恢复。
        self.chunks = json.loads(
            (directory / "chunks.json").read_text(
                encoding="utf-8"
            )
        )

        # 根据加载的向量矩阵，恢复每条向量的维度。
        #
        # 如果矩阵形状为 (3, 1024)，
        # shape[1] 就是 1024。
        #
        # 这样即使创建对象时使用了默认的 dim=1024，
        # 也可以根据保存的数据恢复实际维度。
        #
        # 注意：这只适用于文件本身有效的情况。
        if self.vectors.ndim != 2:
            raise ValueError("加载的向量矩阵必须是二维数组")

        self.dim = self.vectors.shape[1]

        # 检查向量条目数与文本片段数是否一致。
        #
        # 如果一个文件有 3 条向量，另一个文件却有 2 条文本，
        # 就说明数据不完整或两个文件不匹配。
        #
        # 必须保证：
        # len(self.vectors) == len(self.chunks)
        if len(self.vectors) != len(self.chunks):
            raise ValueError(
                f"加载失败：向量数量 {len(self.vectors)} "
                f"与文本块数量 {len(self.chunks)} 不匹配"
            )


    def remove_doc(self,doc_id:str)->int:
        # 注意是 self.chunks（属性），不是 self, chunks（两个参数）
        keep = [i for i,chunk in enumerate(self.chunks) if chunk["doc_id"] != doc_id]
        removed = len(self.chunks) -len(keep)

        if removed:
            self.vectors = self.vectors[keep]
            self.chunks = [self.chunks[i] for i in keep]

        return removed


# ============================================================
# 以下是测试代码。
#
# if __name__ == "__main__":
# 只有当这个 Python 文件被直接运行时，
# 下面的代码才会执行。
#
# 如果其他文件通过 import 导入 VectorStore，
# 下面的测试代码不会自动执行。
#
# 注意：这个判断必须放在类定义外部，
# 不能缩进在 class VectorStore 内部。
# ============================================================

if __name__ == "__main__":

    # 创建随机数生成器，并固定随机种子为 0。
    #
    # default_rng(0)：
    # 每次运行时，使用同样的随机数生成序列。
    #
    # 好处：
    # 测试结果可以复现，便于调试和排查问题。
    #
    # 注意：
    # 这只是生成随机测试数据，不是真正的文本 Embedding。
    rng = np.random.default_rng(0)

    # 创建三个文本片段。
    #
    # 每个字典都包含：
    # chunk_id -> 文本片段的唯一标识
    # text     -> 实际的文本内容
    #
    # 真实 RAG 项目中，还可以包含：
    # source_file -> 来源文件
    # page        -> 页码
    # metadata    -> 其他元数据
    chunks = [
        {
            "chunk_id": "c1",
            "text": "苹果是一种水果"
        },
        {
            "chunk_id": "c2",
            "text": "香蕉是黄色的"
        },
        {
            "chunk_id": "c3",
            "text": "汽车需要汽油"
        },
    ]

    # 生成 3 条随机向量，每条向量有 1024 个数字。
    #
    # rng.normal(size=(3, 1024))：
    # 从标准正态分布中生成随机数。
    #
    # shape = (3, 1024)
    #
    # 第 0 行 -> 对应 chunks[0]，即苹果
    # 第 1 行 -> 对应 chunks[1]，即香蕉
    # 第 2 行 -> 对应 chunks[2]，即汽车
    #
    # astype(np.float32)：
    # 将生成的数值统一转换为 float32。
    vectors = rng.normal(
        size=(3, 1024)
    ).astype(np.float32)

    # 对每一行向量进行 L2 归一化。
    #
    # np.linalg.norm(vectors, axis=1, keepdims=True)：
    #
    # axis=1：
    # 沿着每一行计算向量长度。
    #
    # keepdims=True：
    # 保留计算结果的二维形状，便于广播除法。
    #
    # 原向量形状：
    # (3, 1024)
    #
    # 每行的范数形状：
    # (3, 1)
    #
    # 除法时，(3, 1) 会沿列方向广播到 (3, 1024)。
    #
    # 归一化公式：
    #
    #       v
    # v' = ------
    #      ||v||
    #
    # 归一化后，每条非零向量的长度约等于 1。
    #
    # 为什么这样做？
    # 因为对于单位向量：
    #
    # cos(theta) = (a · b) / (||a|| ||b||)
    #
    # 当 ||a|| = ||b|| = 1 时：
    #
    # cos(theta) = a · b
    #
    # 所以两个单位向量的点积就等于余弦相似度。
    #
    # 这里随机生成的向量也可以归一化，
    # 但归一化并不会让它们拥有真实的语义。
    vectors /= np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    # 创建一个向量库对象。
    #
    # 默认 dim=1024，
    # 与刚才生成的向量维度保持一致。
    store = VectorStore()

    # 将三条向量及其对应的文本片段添加到向量库。
    #
    # 添加后：
    # store.vectors.shape == (3, 1024)
    # len(store.chunks) == 3
    store.add(vectors, chunks)

    # 查看当前向量库中有多少条文本片段。
    #
    # 会自动调用 store.__len__()。
    print("片段数：", len(store))

    # 使用第一条向量作为查询向量进行测试。
    #
    # vectors[0]：
    # 取得第 0 行，也就是第一条 1024 维向量。
    #
    # top_k=2：
    # 只返回最相似的前两条文本片段。
    #
    # 因为查询向量就是第一条存储向量，
    # 并且所有向量都已归一化，
    # 所以第一条向量与自己的点积理论上为 1。
    #
    # 但剩余结果的排序是随机的，
    # 因为其他向量也都是随机生成的。
    results = store.search(vectors[0], top_k=2)

    # 打印检索结果。
    #
    # 结果中的每个元素是：
    # (文本片段字典, 相似度分数)
    #
    # 例如：
    # [
    #     ({"chunk_id": "c1", "text": "苹果是一种水果"}, 1.0),
    #     ({"chunk_id": "c3", "text": "汽车需要汽油"}, 0.05)
    # ]
    #
    # 上面的具体第二条结果仅为格式示例，
    # 实际分数以程序运行结果为准。
    print("检索结果：", results)

    # 将向量库保存到指定目录。
    #
    # 保存后会产生两个文件：
    #
    # /data/index/test_store/vectors.npy
    # /data/index/test_store/chunks.json
    #
    # 如果当前系统没有权限创建 /data/index，
    # 或者指定路径不可写，就会出现文件系统错误。
    store.save("data/index/test_store")

    # 创建一个全新的向量库对象。
    #
    # 此时 loaded 中还没有文本片段，
    # 也没有已经保存的向量。
    loaded = VectorStore()

    # 从磁盘读取刚才保存的数据。
    #
    # 加载完成后：
    # loaded.vectors.shape == (3, 1024)
    # len(loaded.chunks) == 3
    loaded.load("data/index/test_store")

    # 验证重新加载后的文本片段数量。
    print("重新加载后片段数：", len(loaded))

    # 验证重新加载后的检索结果是否正确。
    #
    # loaded.search(vectors[0], top_k=1)：
    # 返回最相似的第一条结果。
    #
    # [0]：
    # 取得第一条检索结果，也就是一个二元组。
    #
    # [0][0]：
    # 取得这个二元组中的文本片段字典。
    #
    # ["chunk_id"]：
    # 读取文本片段的 ID。
    #
    # == "c1"：
    # 检查检索出的第一条文本是否为 c1。
    #
    # 因为查询向量与 c1 的向量完全相同，
    # 所以正常情况下应该返回 True。
    print(
        "加载后检索一致：",
        loaded.search(vectors[0], top_k=1)[0][0]["chunk_id"] == "c1"
    )
