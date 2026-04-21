import argparse
from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import subprocess
import threading
import time
import re
import json
import os
import openai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ================= 配置区域 =================
CONFIG = {
    "yaml_path": r"E:\Mxy\HiSim\agentverse\tasks\simulation\test\config.yaml",
    "task": "simulation/test",
    "ckpt": "path_to_save_the_intermediate_status",
    "raw_json_name": r"E:\Mxy\HiSim\output\temp\test.json",
    "output_dir": "ai_research_json"
}
FILE_MODEL = "influence_bcm/influence_bc_model.json"
FILE_TEST = "E:/Mxy/HiSim/output/temp/test.json"
OUT_HTML = "simulation_dashboard_culture_v4_dynamic.html"
# ================= AI 模型配置 (已切换为 GPT-3.5) =================
# 请在此处填入您的 OpenAI API Key (或者您的智增增等 GPT 代理 Key)
openai.api_key = "sk-zk2d3e3efd833b8863a0b9178b4e54007fe690803ba2eb44"

# 如果您使用的是官方直连，请注释掉下面这行；
# 如果您使用的是代理（比如智增增），请保留或修改为对应的代理地址
# 注意：旧版 openai 库用 openai.api_base，新版用 openai.base_url
openai.api_base = "https://api.zhizengzeng.com/v1"
# =============================================================
# ================= 数据读取与智能清洗 =================
def load_json(path):
    if not os.path.exists(path):
        print(f"⚠️ 警告: 找不到文件 {path}，使用空字典。")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 辅助函数：从小 agent 的 info 字典中提取态度值
def extract_att(info_dict):
    if not isinstance(info_dict, dict):
        return 0.0
    for k, v in info_dict.items():
        # 兼容多种可能出现的态度字段名
        if str(k).lower() in ["attitude", "opinion", "state", "att"]:
            if isinstance(v, list):
                return float(v[0]) if v else 0.0
            try:
                return float(v)
            except:
                pass
    return 0.0


def generate_html():
    raw_model = load_json(FILE_MODEL)
    raw_test = load_json(FILE_TEST)

    # --- 1. 解析动态网络结构 & 小 agent 态度值 ---
    nodes_set = set()
    edges_by_round = {}
    model_atts = {}

    model_round_keys = [int(k) for k in raw_model.keys() if k.isdigit()]

    if not model_round_keys:
        static_edges = []
        model_data = raw_model.get("0", raw_model)
        model_atts[0] = {}
        if isinstance(model_data, dict):
            for target, info in model_data.items():
                nodes_set.add(target)
                model_atts[0][target] = extract_att(info)
                influencers = info.get("influencers", [])
                for src in influencers:
                    nodes_set.add(src)
                    static_edges.append({"from": src, "to": target})
        edges_by_round[0] = static_edges
    else:
        for r in sorted(model_round_keys):
            r_str = str(r)
            round_data = raw_model[r_str]
            current_edges = []
            model_atts[r] = {}
            if isinstance(round_data, dict):
                for target, info in round_data.items():
                    nodes_set.add(target)
                    model_atts[r][target] = extract_att(info)
                    influencers = info.get("influencers", [])
                    for src in influencers:
                        nodes_set.add(src)
                        current_edges.append({"from": src, "to": target})
            edges_by_round[r] = current_edges

    all_nodes = sorted(list(nodes_set))

    # --- 2. 核心用户判定 ---
    import re
    normal_pattern = re.compile(r"^user_\d+$")
    core_users = [u for u in all_nodes if not normal_pattern.match(u)]

    # --- 3. 解析 test.json 中的核心大V数据 ---
    parsed_atts = {}
    parsed_actions = {}
    test_round_keys = []

    for turn_key, turn_data in raw_test.items():
        if turn_key.startswith("turn_"):
            try:
                r = int(turn_key.split("_")[1])
                test_round_keys.append(r)
                parsed_atts[r] = {}
                parsed_actions[r] = {}

                agents_data = turn_data.get("agents", {})
                for agent, info in agents_data.items():
                    raw_att = info.get("attitude", 0.0)
                    parsed_atts[r][agent] = float(raw_att[0]) if isinstance(raw_att, list) else float(raw_att)

                    parsed_actions[r][agent] = {
                        "action": info.get("action", ""),
                        "content": info.get("content", "Silence")
                    }
            except ValueError:
                continue

    # --- 4. 动态确定最大公约轮次 ---
    model_max = max(model_round_keys) if model_round_keys else 0
    test_max = max(test_round_keys) if test_round_keys else 0

    # 【修复1】：从 min 改为 max，防止静态拓扑图锁死轮次推演
    target_max_input = max(model_max, test_max)
    print(f"📊 自动检测: Model({model_max}), TestJson({test_max})")
    print(f"✅ 自动设定最高进度为第 {target_max_input} 轮 (UI将显示至 Cycle {target_max_input + 1})")

    vis_atts = {}
    vis_actions = {}
    vis_edges = {}

    # --- 5. 双重数据融合 (核心逻辑) ---
    # 【修复2】：统一采用 0-based 索引，解决 Python 字典与 JS 取值的错位问题
    for r in range(target_max_input + 1):
        vis_atts[r] = {}

        # A. 继承上一轮的状态作为兜底
        if r > 0:
            vis_atts[r].update(vis_atts[r - 1])
        else:
            vis_atts[r] = {u: 0.0 for u in all_nodes}

        # B. 覆盖小 agent 的态度
        if r in model_atts:
            vis_atts[r].update(model_atts[r])

        # C. 覆盖大 V 的最新态度
        if r in parsed_atts:
            vis_atts[r].update(parsed_atts[r])

        # 行为同步
        if r in parsed_actions:
            vis_actions[r] = parsed_actions[r]
        else:
            vis_actions[r] = {}

        # 拓扑同步
        if r in edges_by_round:
            vis_edges[r] = edges_by_round[r]
        else:
            vis_edges[r] = vis_edges.get(r - 1, [])

    max_vis_round = target_max_input

    # ================= HTML 渲染部分 =================
    html_template = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>CultureFlow Analytics</title>
<script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&family=Noto+Serif+SC:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>

<style>
  :root {
    --bg-color: #F9F7F2; 
    --card-bg: #FFFFFF;
    --text-main: #2C2420;
    --text-sub: #6B6560;
    --accent-red: #8B1E1E;
    --border-color: #E5E0D8;
    --shadow-soft: 0 4px 20px -2px rgba(44, 36, 32, 0.08);
  }

  body { 
    background-color: var(--bg-color); 
    background-image: radial-gradient(circle at 10% 10%, #fff 0%, transparent 80%);
    color: var(--text-main); 
    font-family: 'Inter', system-ui, sans-serif; 
    overflow: hidden; 
  }

  .serif-font { font-family: 'Playfair Display', 'Noto Serif SC', serif; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #D1CEC7; border-radius: 3px; }

  div.vis-tooltip {
    background-color: var(--card-bg);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 8px 12px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    box-shadow: var(--shadow-soft);
    border-radius: 6px;
    z-index: 1000;
  }

  .paper-card {
    background: var(--card-bg);
    border-right: 1px solid var(--border-color);
    box-shadow: var(--shadow-soft);
  }

  .control-card {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-soft);
    border-radius: 12px;
  }

  input[type=range] { -webkit-appearance: none; background: transparent; }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; height: 16px; width: 16px;
    background: var(--accent-red); cursor: pointer; margin-top: -6px;
    border-radius: 50%;
    border: 2px solid #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }
  input[type=range]::-webkit-slider-runnable-track { 
    width: 100%; height: 4px; 
    background: #E0DDD5; border-radius: 2px;
  }

  .btn-primary {
    background-color: var(--text-main);
    color: #fff;
    transition: all 0.2s;
  }
  .btn-primary:hover {
    background-color: var(--accent-red);
    transform: translateY(-1px);
  }
</style>
</head>
<body class="flex h-screen w-screen relative">

  <div class="absolute inset-0 z-0 pointer-events-none opacity-40" 
       style="background-image: url('data:image/svg+xml,%3Csvg width=\\'60\\' height=\\'60\\' viewBox=\\'0 0 60 60\\' xmlns=\\'http://www.w3.org/2000/svg\\'%3E%3Cg fill=\\'none\\' fill-rule=\\'evenodd\\'%3E%3Cg fill=\\'%239C92AC\\' fill-opacity=\\'0.05\\'%3E%3Cpath d=\\'M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z\\'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E');">
  </div>

  <div class="w-[380px] h-full paper-card flex flex-col z-20 transition-transform duration-300 translate-x-0 relative" id="sidePanel">
    <div class="p-6 border-b border-[#E5E0D8]">
      <div class="flex items-center gap-2 mb-4">
          <div class="w-2 h-2 rounded-full bg-[#8B1E1E]"></div>
          <div class="w-1.5 h-1.5 rounded-full bg-[#2C2420]"></div>
          <span class="serif-font text-lg font-bold tracking-wide text-[#2C2420] ml-1">CultureFlow</span>
      </div>

      <div class="text-[10px] text-[#8B1E1E] uppercase tracking-widest font-bold mb-1 opacity-80">System Insight</div>
      <h2 id="panelTitle" class="text-3xl serif-font text-[#2C2420]">Awaiting Input</h2>
      <div id="panelMeta" class="mt-3 text-sm text-gray-500 font-light italic">Select a node to view cognitive trace.</div>
    </div>

    <div class="flex-1 overflow-y-auto bg-[#FAFAF9]" id="panelContent">
        <div class="p-10 text-gray-400 text-sm font-light text-center mt-10 serif-font">
            <span class="block text-2xl mb-2 text-[#E5E0D8] display-block">❧</span>
            Select a node from the network<br>to analyze simulation data.
        </div>
    </div>

    <div class="p-3 bg-white text-[10px] text-center text-gray-400 border-t border-gray-100 uppercase tracking-widest">
        Analysis Module v2.1 (Action Only)
    </div>
  </div>

  <div class="relative flex-1 h-full z-10 flex flex-col">
    <div id="mynetwork" class="w-full h-full cursor-grab active:cursor-grabbing"></div>

    <div class="absolute top-6 right-8 pointer-events-none text-right">
        <h1 class="text-2xl font-bold text-[#2C2420] serif-font">Dynamic Graph</h1>
        <div class="text-xs text-gray-400 mt-1 uppercase tracking-widest">Round-based Topology</div>
    </div>

    <div class="absolute bottom-8 left-10 right-10 control-card p-4 flex items-center gap-6">
      <button id="playBtn" class="serif-font btn-primary px-8 py-2.5 rounded-full text-sm shadow-md font-semibold tracking-wide min-w-[120px]">
        INITIATE
      </button>

      <div class="flex-1 flex flex-col px-4">
        <div class="flex justify-between text-xs font-semibold text-[#6B6560] mb-2 uppercase tracking-wider">
          <span class="flex items-center gap-2">
            Simulation Round 
            <span id="roundDisplay" class="bg-[#2C2420] text-white px-2 py-0.5 rounded text-xs font-mono">1</span>
          </span>
          <span>Max: {MAX_VIS_ROUND_DISPLAY}</span>
        </div>
        <input type="range" id="roundSlider" min="0" max="{MAX_VIS_ROUND}" value="0" class="w-full">
      </div>
    </div>
  </div>

<script>
  const rawAtts = {JSON_ATTS};
  const rawActions = {JSON_ACTIONS};
  const rawEdges = {JSON_EDGES}; 
  const allNodes = {JSON_ALL_NODES};
  const coreUsers = new Set({JSON_CORE_USERS});
  const maxRound = {MAX_VIS_ROUND};

  function getAttitudeColor(att) {
    const val = Math.max(-1, Math.min(1, att));
    if (Math.abs(val) < 0.01) return '#9CA3AF';
    if (val > 0) {
        const opacity = 0.3 + 0.7 * Math.abs(val);
        return `rgba(20, 148, 133, ${opacity})`; 
    }
    const opacity = 0.3 + 0.7 * Math.abs(val);
    return `rgba(168, 45, 45, ${opacity})`; 
  }

  function getEdgeColor(sourceAtt) {
    const val = sourceAtt || 0;
    if (Math.abs(val) < 0.01) return { color: '#D1D5DB', opacity: 0.4, width: 1 };
    const mag = Math.min(1, Math.abs(val));
    if (val > 0) return { color: '#2A9D8F', opacity: 0.3 + 0.5 * mag, width: 1 + 1.5 * mag };
    return { color: '#E76F51', opacity: 0.3 + 0.5 * mag, width: 1 + 1.5 * mag };
  }

  function getAvatar(seed) {
    return `https://api.dicebear.com/7.x/avataaars/png?seed=${seed}&size=128&backgroundColor=transparent`;
  }

  const nodes = new vis.DataSet();
  const edges = new vis.DataSet(); 

  allNodes.forEach((u, index) => {
    const isCore = coreUsers.has(u);
    let size = isCore ? 50 : 15; 

    nodes.add({
      id: u,
      label: isCore ? u : '',
      shape: isCore ? 'circularImage' : 'dot',
      image: isCore ? 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7' : undefined,
      size: size,
      borderWidth: isCore ? 4 : 1, 
      shapeProperties: { useBorderWithImage: true, useImageSize: false },
      font: { color: '#2C2420', face: 'Inter', size: 14, background: 'rgba(255,255,255,0.7)', strokeWidth: 0 },
      shadow: { enabled: true, color: 'rgba(0,0,0,0.1)', size: 10, x: 2, y: 2 }
    });

    if (isCore) {
      setTimeout(() => {
        nodes.update({
          id: u,
          image: getAvatar(u),
          brokenImage: 'https://api.dicebear.com/7.x/initials/png?seed=' + u + '&backgroundColor=e5e5e5'
        });
      }, index * 150);
    }
  });

  const container = document.getElementById('mynetwork');
  const network = new vis.Network(container, { nodes, edges }, {
    nodes: { 
        borderWidth: 1, 
        borderWidthSelected: 6, 
        color: { border: '#6B7280', background: '#9CA3AF' },
        chosen: {
            node: function(values, id, selected, hovering) {
                if (selected) {
                    values.size = values.size * 1.5;
                    values.borderColor = '#D97706';
                } else if (hovering) {
                    values.size = values.size * 1.1;
                    values.borderColor = '#D97706';
                }
            }
        }
    },
    edges: { 
        smooth: { type: 'continuous', roundness: 0.2 }, 
        arrows: { to: { enabled: true, scaleFactor: 0.5 } }, 
        color: { inherit: false } 
    },
    physics: { 
        forceAtlas2Based: { 
            gravitationalConstant: -30, 
            centralGravity: 0.005, 
            springLength: 120, 
            springConstant: 0.04, 
            damping: 0.4 
        }, 
        solver: 'forceAtlas2Based', 
        stabilization: false 
    },
    interaction: { hover: true, tooltipDelay: 100, hideEdgesOnDrag: true }
  });

  let currentRound = 0;
  let selectedNodeId = null;
  let isPlaying = false;
  let playInterval = null;

  function updateVisualization(round) {
      const curAtts = rawAtts[round] || {};
      const curEdges = rawEdges[round] || []; 

      const nodeUpdates = [];
      allNodes.forEach(u => {
        const att = curAtts[u] || 0;
        const colorStyle = getAttitudeColor(att);
        const isCore = coreUsers.has(u);

        if (isCore) {
          nodeUpdates.push({ id: u, color: { border: colorStyle, background: '#FFFFFF' } });
        } else {
          nodeUpdates.push({ id: u, color: { background: colorStyle, border: '#fff' } });
        }
      });
      nodes.update(nodeUpdates);

      const newEdgeData = curEdges.map((e, i) => {
          const sourceAtt = curAtts[e.from] || 0; 
          const style = getEdgeColor(sourceAtt);
          return {
              from: e.from, to: e.to,
              color: { color: style.color, opacity: style.opacity },
              width: style.width
          };
      });

      edges.clear();
      edges.add(newEdgeData);

      if (selectedNodeId) renderSidebar(selectedNodeId);

      const content = document.getElementById('panelContent');
      content.scrollTop = content.scrollHeight;
  }

  window.onload = function() {
      const content = document.getElementById('panelContent');
      content.scrollTop = content.scrollHeight; 
  }

  function renderSidebar(nodeId) {
    const title = document.getElementById('panelTitle');
    const meta = document.getElementById('panelMeta');
    const content = document.getElementById('panelContent');
    const att = (rawAtts[currentRound] || {})[nodeId] || 0;

    title.innerText = nodeId;
    const isCore = coreUsers.has(nodeId);

    let attColorClass = 'text-gray-500';
    if (att > 0) attColorClass = 'text-[#2A9D8F]';
    else if (att < 0) attColorClass = 'text-[#E76F51]';

    meta.innerHTML = `
      <div class="flex items-center gap-4 mt-2">
         <span class="px-2 py-1 bg-[#F5F5F4] text-[#78716C] text-xs font-semibold rounded border border-[#E7E5E4] uppercase tracking-wider">${isCore ? 'Core Entity' : 'Network Node'}</span>
         <div class="flex items-center gap-2 text-sm font-medium">
            <span class="text-gray-400 font-serif italic">Stance:</span>
            <span class="${attColorClass}">${att.toFixed(3)}</span>
         </div>
      </div>
    `;

    let html = '';
    if (isCore) {
      let hasData = false;
      html += '<div class="flex flex-col gap-6 p-6">'; 
      for (let r = 0; r <= currentRound; r++) {
        const roundData = rawActions[r];
        if (roundData && roundData[nodeId]) {
            hasData = true;
            const act = roundData[nodeId];

            const rawContent = act.content || "Silence";
            const rawAction = act.action || "No action recorded.";

            let displayHtml = "";
            if (rawContent === "Silence") {
                displayHtml = `<span class="italic text-gray-400">[Action Log]</span><br>${rawAction}`;
            } else {
                displayHtml = rawContent.replace(/\\n/g, '<br>');
            }

            html += `
            <div class="relative group">
                <div class="absolute -left-[29px] top-2 w-1.5 h-1.5 bg-[#D6D3D1] rounded-full ring-4 ring-[#FAFAF9]"></div>
                <div class="text-[10px] text-gray-400 font-bold mb-1.5 flex items-center justify-between uppercase tracking-wider"><span>Round ${r + 1}</span></div>
                <div class="bg-white rounded-lg shadow-sm border border-[#E5E0D8] overflow-hidden transition hover:shadow-md">
                    <div class="p-4 text-[#2C2420] text-sm leading-relaxed font-sans border-l-4 border-[#2C2420]">${displayHtml}</div>
                </div>
            </div>`;
        }
      }
      html += '</div>';
      if (!hasData) { html = '<div class="h-full flex flex-col items-center justify-center text-gray-400 text-xs italic font-serif opacity-60 mt-10">No activity logged.</div>'; }
    } else {
      html = '<div class="p-8 text-center text-gray-400 text-xs border-t border-[#E5E0D8] bg-[#FAFAF9]">Standard nodes propagate influence but do not emit linguistic logs.</div>';
    }

    content.innerHTML = isCore ? '<div class="absolute left-6 top-0 bottom-0 w-px bg-[#E5E0D8] z-0"></div>' + html : html;
    setTimeout(() => { content.scrollTop = content.scrollHeight; }, 50);
  }

  const slider = document.getElementById('roundSlider');
  const roundDisplay = document.getElementById('roundDisplay');
  const playBtn = document.getElementById('playBtn');

  slider.addEventListener('input', (e) => {
    currentRound = parseInt(e.target.value);
    roundDisplay.innerText = currentRound + 1; // UI显示+1
    updateVisualization(currentRound);
  });

  playBtn.addEventListener('click', () => {
    if (isPlaying) {
        clearInterval(playInterval);
        playBtn.innerText = "INITIATE";
        playBtn.style.backgroundColor = "var(--text-main)";
    } else {
        playBtn.innerText = "HALT";
        playBtn.style.backgroundColor = "var(--accent-red)";
        playInterval = setInterval(() => {
            if (currentRound < maxRound) {
                currentRound++;
                slider.value = currentRound;
                roundDisplay.innerText = currentRound + 1; // UI显示+1
                updateVisualization(currentRound);
            } else { playBtn.click(); }
        }, 800);
    }
    isPlaying = !isPlaying;
  });

  network.on("click", function (params) {
    if (params.nodes.length > 0) {
      selectedNodeId = params.nodes[0];
      renderSidebar(selectedNodeId);
    } else {
      selectedNodeId = null;
      document.getElementById('panelTitle').innerText = "Awaiting Input";
      document.getElementById('panelContent').innerHTML = '<div class="p-10 text-gray-400 text-sm font-light text-center mt-10 serif-font"><span class="block text-2xl mb-2 text-[#E5E0D8]">❧</span>Select a node to analyze data.</div>';
    }
  });

  updateVisualization(0);
</script>
</body>
</html>
"""

    html_content = html_template.replace("{MAX_VIS_ROUND_DISPLAY}", str(max_vis_round + 1))
    html_content = html_content.replace("{MAX_VIS_ROUND}", str(max_vis_round))
    html_content = html_content.replace("{JSON_ATTS}", json.dumps(vis_atts))
    html_content = html_content.replace("{JSON_ACTIONS}", json.dumps(vis_actions))
    html_content = html_content.replace("{JSON_EDGES}", json.dumps(vis_edges))
    html_content = html_content.replace("{JSON_ALL_NODES}", json.dumps(all_nodes))
    html_content = html_content.replace("{JSON_CORE_USERS}", json.dumps(core_users))

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ 生成完毕: {OUT_HTML} (数据源已修复同步对齐！)")

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

GLOBAL_STATE = {
    "current_suffix": "",
    "current_event": ""
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ----------------- 数据处理工具函数 -----------------
def run_simulation():
    cmd = [
        "agentverse-simulation",
        "--task", CONFIG["task"],
        "--ckpt", CONFIG["ckpt"],
    ]
    subprocess.Popen(cmd)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------- 数据提取逻辑 -----------------
def extract_data_logic(raw_input_path, action_output_path, att_output_path):
    try:
        if not os.path.exists(raw_input_path):
            print(f"⚠️ Waiting for raw file: {raw_input_path}")
            return

        with open(raw_input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_user_actions = {}
        att = {}
        pattern = r"content=(.*?)\s+sender="

        for user_name, rounds in data.items():
            if user_name == 'opinion_results':
                att = rounds
                continue
            if not isinstance(rounds, dict):
                continue

            all_user_actions[user_name] = {}
            for round_id, round_info in rounds.items():
                if not isinstance(round_info, dict): continue
                text = round_info.get('response', '')
                parse_response = round_info.get('parsed_response', [])
                thought = parse_response[1] if isinstance(parse_response, list) and len(parse_response) > 1 else None
                match = re.search(pattern, text, re.S)
                action = match.group(1).strip() if match else "Silence"
                if not action: action = "Silence"

                all_user_actions[user_name][round_id] = {
                    "action": action,
                    "thought": thought
                }

        with open(action_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_user_actions, f, ensure_ascii=False, indent=4)
        with open(att_output_path, 'w', encoding='utf-8') as f:
            json.dump(att, f, ensure_ascii=False, indent=4)
        print(f"🔄 Data extracted to {action_output_path}")

    except Exception as e:
        print(f"❌ Extraction Error: {str(e)}")


def background_extraction_loop(suffix):
    raw_path = CONFIG["raw_json_name"]
    action_out = os.path.join(CONFIG["output_dir"], f"actions_test.json")
    att_out = os.path.join(CONFIG["output_dir"], f"atts_test.json")
    print(f"⏰ Background loop started: watching {raw_path}")

    # 记录上一次检测到的【已完成完整生成的轮次数】
    last_completed_rounds = 0

    # 定义一轮需要集齐的 Agent 数量
    EXPECTED_AGENTS_PER_ROUND = 50

    while True:
        # 1. 运行原有的抽取逻辑
        extract_data_logic(raw_path, action_out, att_out)

        # 2. 【精准检测】根据 Agent 数量判断是否生成完毕了一轮
        if os.path.exists(raw_path):
            try:
                # 读取当前的 json 数据
                with open(raw_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 统计当前总共有多少个【完整轮次】
                current_completed_rounds = 0

                for turn_key, turn_data in data.items():
                    if turn_key.startswith("turn_"):
                        agents_dict = turn_data.get("agents", {})
                        # 如果这一轮的 agent 数量达到了 50 个，说明本轮已经全部生成完毕
                        if len(agents_dict) >= EXPECTED_AGENTS_PER_ROUND:
                            current_completed_rounds += 1

                # 如果完整的轮次数增加了，说明刚刚有新的一轮闭环了！
                if current_completed_rounds > last_completed_rounds:
                    print(
                        f"🔄 检测到第 {current_completed_rounds} 轮的 {EXPECTED_AGENTS_PER_ROUND} 个 Agent 已全部生成完毕！")
                    print("⚙️ 正在后台动态生成最新网络拓扑图 HTML...")

                    # =========== 调用网络图生成函数 ===========
                    generate_html()
                    # ==========================================

                    last_completed_rounds = current_completed_rounds
                    print("✅ 最新网络拓扑图 HTML 已自动生成就绪！")

            except json.JSONDecodeError:
                # 刚好在写入文件时读取可能会报 JSON 解析错误，直接忽略，等下个 20 秒周期再读即可
                pass
            except Exception as e:
                print(f"❌ 自动生成网络图 HTML 时出错: {str(e)}")

        # 挂起 20 秒后再次检查
        time.sleep(20)


def update_yaml(yaml_path, max_turns, current_time, time_delta, trigger_news):
    path = Path(yaml_path)
    if not path.exists(): return
    yaml = YAML()
    yaml.preserve_quotes = True
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f)
    env = data.setdefault("environment", CommentedMap())
    env["max_turns"] = int(max_turns)
    env["current_time"] = str(current_time)
    env["time_delta"] = int(time_delta)
    trigger_map = env.get("trigger_news")
    if not isinstance(trigger_map, CommentedMap):
        trigger_map = CommentedMap()
        env["trigger_news"] = trigger_map
    trigger_map.clear()
    trigger_map[0] = trigger_news
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


# ================= AI 洞察上下文提取 =================
def get_latest_simulation_context():
    json_path = "./output/temp/test.json"
    if not os.path.exists(json_path):
        return "当前暂无推演数据（test.json 未生成）。"

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        turns = sorted([k for k in data.keys() if k.startswith('turn_')], key=lambda x: int(x.split('_')[1]))
        if not turns:
            return "推演数据为空。"

        recent_turns = turns[-3:]

        context_lines = []
        for turn in recent_turns:
            round_num = int(turn.split('_')[1]) + 1
            context_lines.append(f"--- 第 {round_num} 轮 (Round {round_num}) ---")
            agents = data[turn].get("agents", {})
            for agent, info in agents.items():
                raw_att = info.get("attitude", 0.0)
                attitude = float(raw_att[0]) if isinstance(raw_att, list) else float(raw_att)
                content = info.get("content", "Silence")
                action = info.get("action", "")
                context_lines.append(
                    f"人物: {agent} | 态度值(>0支持,<0反对): {attitude:.2f} | 发言内容: {content} | 隐藏动作: {action}")

        return "\n".join(context_lines)

    except Exception as e:
        return f"读取数据失败: {str(e)}"


# ----------------- 路由定义 -----------------
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/start', methods=['POST'])
def handle_start():
    data = request.get_json(force=True) or {}
    trigger_news_raw = data.get('trigger_news', '')
    trigger_news = ' '.join(str(trigger_news_raw).split())

    clean_news = re.sub(r'[^\w\s]', '', trigger_news)
    words = clean_news.split()
    suffix = f"{words[-2]}_{words[-1]}" if len(words) >= 2 else "default_event"

    GLOBAL_STATE["current_suffix"] = suffix
    GLOBAL_STATE["current_event"] = trigger_news

    update_yaml(
        CONFIG["yaml_path"],
        data.get('max_turns'),
        data.get('current_time'),
        data.get('time_delta'),
        trigger_news
    )

    try:
        sim_thread = threading.Thread(target=run_simulation, daemon=True)
        sim_thread.start()
    except NameError:
        print("⚠️ run_simulation 函数未定义，暂未启动仿真进程。")

    bg_thread = threading.Thread(
        target=background_extraction_loop,
        args=(suffix,),
        daemon=True
    )
    bg_thread.start()

    return jsonify({
        "status": "success",
        "file_suffix": suffix,
        "trigger_news": trigger_news,
        "message": "Simulation started."
    })


# ================= AI 实时对话洞察接口 (已指定 GPT-3.5) =================
@app.route('/ask_insight', methods=['POST'])
def handle_ask_insight():
    data = request.get_json(force=True) or {}
    question = data.get('question', '').strip()

    if not question:
        return jsonify({"answer": "我没有收到您的问题，请重新输入。"})

    event_desc = GLOBAL_STATE.get("current_event") or "默认推演事件（尚未设定）"
    context_data = get_latest_simulation_context()

    system_prompt = "You are an expert in computational social science and cultural communication. You act as an AI Assistant for a social simulation dashboard. Your language should be professional, objective, and strictly in Chinese."

    user_prompt = f"""
    【系统背景】
    当前正在推演的文化传播事件是：
    "{event_desc}"

    【最新推演数据摘要（最近3轮）】
    {context_data}

    【用户提问】
    {question}

    请结合上述“推演事件背景”和“最新推演数据摘要”，用专业、精炼的中文回答用户的提问。如果用户的提问与数据无关，请直接解答其问题，但可适当结合传播学视角。
    回答请直接输出结论，无需复述原有的数据。
    """

    print("🧠 向 GPT-3.5 发送洞察分析请求...")
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # <--- 已修改为 GPT-3.5 模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )

        # 兼容不同版本的 openai 库返回格式
        if isinstance(completion, dict):
            answer = completion["choices"][0]["message"]["content"]
        else:
            answer = completion.choices[0].message.content

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"❌ 大模型调用失败: {str(e)}")
        return jsonify({"answer": f"抱歉，连接 AI 引擎失败，错误信息：{str(e)}"})


@app.route('/generated_data/<path:filename>')
def serve_data(filename):
    return send_from_directory(CONFIG["output_dir"], filename)


if __name__ == "__main__":
    print("🌍 Server running at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)