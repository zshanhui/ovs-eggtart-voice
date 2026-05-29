#!/usr/bin/env python3
"""Convert a HuggingFace chat model to .rkllm format for Rockchip NPUs.

Usage (on a CUDA PC — Vast.ai, Lambda, or local GPU workstation)::

    python3 build_rkllm_model.py \\
        --model Qwen/Qwen3-0.6B \\
        --target RK3576 \\
        --quant W4A16

Output: ``qwen3-0.6b-instruct_W4A16_RK3576.rkllm``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def _calibration_data(model_name: str) -> list[dict[str, str]]:
    """~120 mixed zh/en examples for quantisation calibration.

    Covers three domains in the ChatML format:

    1. **General voice assistant** (~35 examples) — the system prompt from
       ``agent/apps/rk3576-chat/config.yaml``.  Conversational zh+en, short,
       voice-style inputs.

    2. **Restaurant AI assistant** (~40 examples) — ordering, menu questions,
       dietary needs, reservations, complaints, payment.  Customer-facing
       voice assistant in a smart restaurant.

    3. **Chef / KDS** (~40 examples) — Kitchen Display System for back-of-house.
       Order queue management, dish prep timing, customer arrival sync,
       inventory and stock alerts, order modifications, rush prioritisation,
       and food safety checks.  zh + en.
    """

    # ── System prompts (varied so quantiser sees different prefixes) ────
    sys_general = "你是一个运行在瑞芯微芯片上的语音助手。请用简洁自然的中文回答，控制在三句话以内，就像和朋友聊天一样。"
    sys_resto_zh = "你是一家智能餐厅的语音助手，负责帮客人点菜、推荐菜品、处理预订和回答疑问。请用简洁友好的中文回答，每次控制在两到三句话。"
    sys_resto_en = "You are a voice assistant at a smart restaurant. Help customers order food, answer menu questions, handle reservations, and respond to requests. Keep replies short and friendly — two to three sentences max."
    sys_chef_zh = "你是一套智能厨房显示系统（KDS），帮助厨师和后厨团队管理订单、安排出菜顺序、监控库存和协调备餐流程。请用简洁专业的厨房用语回答，每次控制在两到三句话，必要时列出条目。"
    sys_chef_en = "You are a smart Kitchen Display System (KDS) helping chefs and kitchen staff manage orders, prioritize dishes, track inventory, and coordinate prep. Respond in short, professional kitchen language — two to three sentences max. Use bullet points when listing."
    sys_generic_en = "You are a helpful assistant."

    def _prompt(system: str) -> str:
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"

    p_gen = _prompt(sys_general)
    p_rz = _prompt(sys_resto_zh)
    p_re = _prompt(sys_resto_en)
    p_kz = _prompt(sys_chef_zh)
    p_ke = _prompt(sys_chef_en)
    p_ge = _prompt(sys_generic_en)

    examples = [
        # =================================================================
        # A. GENERAL VOICE ASSISTANT (40 examples, sys_general)
        # =================================================================

        # ── Greetings & small talk ─────────────────────────────────────
        {"input": p_gen + "你好！\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "你好！有什么我可以帮你的吗？"},
        {"input": p_gen + "嗨\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "嗨！有什么想问的尽管说。"},
        {"input": p_gen + "今天天气怎么样？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "这个我暂时查不到实时天气，不过建议你看看窗外或者打开天气App确认一下。"},
        {"input": p_gen + "你好，今天过得怎么样？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "哈哈，我可是芯片驱动的，心情一直很稳定！你呢，今天顺利吗？"},
        {"input": p_gen + "你叫什么名字？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "我是在瑞芯微NPU上运行的语音助手，没有专门的名字，你就叫我助手吧。"},

        # ── Q&A — zh ──────────────────────────────────────────────────
        {"input": p_gen + "推荐一道简单的家常菜\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "试试西红柿炒鸡蛋吧！两个西红柿配三个鸡蛋，先炒蛋盛出，再炒西红柿，最后混合调味，五分钟搞定。"},
        {"input": p_gen + "树上有5只鸟，飞走了3只，又来了2只，现在有几只？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "现在有4只。5减3剩2，再加2就是4只。"},
        {"input": p_gen + "简单介绍一下人工智能\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "人工智能就是让计算机模仿人类的学习和思考能力，像语音识别、图像理解和对话系统都属于AI的范畴。"},
        {"input": p_gen + "讲个笑话给我听\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "为什么程序员不喜欢户外活动？因为外面阳光里有太多bug！"},
        {"input": p_gen + "如何提高睡眠质量？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "规律作息最重要，睡前别玩手机，房间保持安静和黑暗，白天适当运动也有帮助。"},
        {"input": p_gen + "解释一下什么是物联网\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "物联网就是让日常设备连上网络，互相通信。比如你家的灯、冰箱、空调都能通过手机来控制，这就是物联网的应用。"},
        {"input": p_gen + "今天适合出门吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "我暂时查不到实时天气，不过一般来说，晴天出门比较舒服，雨天记得带伞。"},
        {"input": p_gen + "给我讲个短故事\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "以前有只小狐狸很喜欢帮助别人。有一天小兔子迷路了，小狐狸带它穿过森林找到回家的路。帮助别人自己也很开心。"},
        {"input": p_gen + "你会说英文吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "会的，中英文都可以，你随便切换就行。"},
        {"input": p_gen + "列出三个健康的早餐\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "燕麦粥加水果、全麦面包配鸡蛋、酸奶加坚果和蜂蜜，这三样都很健康。"},
        {"input": p_gen + "我想学编程，从哪里开始？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "建议从Python开始，语法简单，上手快。找一本入门书或者网上免费教程跟着做就可以了。"},

        # ── Q&A — en ──────────────────────────────────────────────────
        {"input": p_gen + "What is the capital of France?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "The capital of France is Paris."},
        {"input": p_gen + "How many continents are there?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "There are seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."},
        {"input": p_gen + "What's your favorite color?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "I don't really have preferences, but I hear blue is pretty popular!"},
        {"input": p_gen + "Tell me a fun fact.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Honey never spoils! Archaeologists found pots of honey in ancient Egyptian tombs over 3,000 years old that are still edible."},
        {"input": p_gen + "Can you speak Chinese?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Yes, I can speak both Chinese and English. Feel free to switch anytime."},

        # ── Commands & acknowledgements ───────────────────────────────
        {"input": p_gen + "好的，谢谢\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "不客气！有需要随时叫我。"},
        {"input": p_gen + "不用了，就这样吧\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "好的，那先这样。有需要再找我。"},
        {"input": p_gen + "等一下\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "没问题，我等你。想好了直接说就行。"},
        {"input": p_gen + "不对，我说错了\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "没关系，重新说就好，我在听。"},
        {"input": p_gen + "能再说一遍吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "当然可以！我刚才说……有什么需要我重复的吗？"},
        {"input": p_gen + "帮我查一下\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "好的，你想查什么呢？给我多一点信息我帮你看看。"},

        # ── Short / casual (voice-style, drop-in queries) ────────────
        {"input": p_gen + "饿了\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "那赶紧去找点吃的吧！想吃什么类型的？我可以帮你推荐。"},
        {"input": p_gen + "好无聊\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "要不我给你推荐部电影或者讲个笑话？无聊的时候找点事做就对了。"},
        {"input": p_gen + "嗯嗯\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "有什么想说的直接说，我听着呢。"},
        {"input": p_gen + "太热了\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "确实，夏天到了。多喝水，开空调，少吃辛辣的，会舒服一点。"},
        {"input": p_gen + "心情不好\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "怎么啦？说出来也许会好受一点。或者给你讲个笑话？"},

        # ── Generic English (sys_generic_en) for diversity ────────────
        {"input": p_ge + "Hello!\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Hello! How can I help you today?"},
        {"input": p_ge + "What time is it?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "I'm sorry, I can't check the current time. You'd need to look at a clock or your device for that."},
        {"input": p_ge + "How do you say 'thank you' in Chinese?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "You say 谢谢 (xiè xiè) for 'thank you' in Chinese."},
        {"input": p_ge + "Write a short introduction about yourself.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "I'm an AI assistant running on a Rockchip NPU, designed to help with questions and tasks through voice interaction."},
        {"input": p_ge + "What is 15 multiplied by 7?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "15 multiplied by 7 equals 105."},

        # =================================================================
        # B. RESTAURANT AI ASSISTANT (40 examples, sys_resto_zh + sys_resto_en)
        # =================================================================

        # ── Ordering — zh ─────────────────────────────────────────────
        {"input": p_rz + "我想点一份宫保鸡丁\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "好的，一份宫保鸡丁，已加入订单。还要加别的吗？"},
        {"input": p_rz + "我要一碗牛肉面和一杯冰红茶\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "收到了，牛肉面加冰红茶。牛肉面要辣的还是不辣的？"},
        {"input": p_rz + "帮我来一份麻婆豆腐和米饭\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "麻婆豆腐配米饭，很经典的搭配。马上帮你下单。"},
        {"input": p_rz + "再加一个番茄蛋汤\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "好的，番茄蛋汤加上了。你的菜快齐了，还需要什么吗？"},
        {"input": p_rz + "来一份炒青菜，不要蒜\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "没问题，炒青菜去蒜，我已经备注好了。"},

        # ── Ordering — en ─────────────────────────────────────────────
        {"input": p_re + "I'd like to order the kung pao chicken, please.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Kung pao chicken, got it. Would you like rice with that?"},
        {"input": p_re + "Can I get a bowl of beef noodle soup?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Absolutely, one beef noodle soup coming up. Mild or spicy?"},
        {"input": p_re + "I'll have the mapo tofu and a side of steamed rice.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Mapo tofu with steamed rice — a classic combo. I'll add that to your order."},
        {"input": p_re + "What vegetarian options do you have?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "We have stir-fried seasonal greens, mapo tofu, tomato and egg soup, and vegetable fried rice. All very popular."},

        # ── Menu questions — zh ────────────────────────────────────────
        {"input": p_rz + "你们有什么招牌菜？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "我们的招牌菜有宫保鸡丁、水煮鱼和回锅肉，都是客人点得最多的。"},
        {"input": p_rz + "有没有不辣的菜？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "有的，番茄炒蛋、清蒸鱼、炒时蔬和蛋花汤都不辣，很适合不吃辣的朋友。"},
        {"input": p_rz + "这个菜里面有什么配料？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "你说的是哪道菜？告诉我菜名我帮你查一下配料表。"},
        {"input": p_rz + "大份和小份差多少？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "大份大概多一半的量，两个人吃的话点大份比较划算。一个人吃小份就够了。"},
        {"input": p_rz + "你们的汤有哪些？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "我们有番茄蛋汤、酸辣汤、紫菜蛋花汤和玉米排骨汤，都是现做的。"},

        # ── Menu questions — en ────────────────────────────────────────
        {"input": p_re + "What's the most popular dish here?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "The kung pao chicken and boiled fish in chili oil are our bestsellers. The twice-cooked pork is also a hit."},
        {"input": p_re + "Is the hot and sour soup very spicy?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "It has a moderate kick but nothing overwhelming. If you're sensitive to spice, I'd recommend the tomato and egg soup instead."},

        # ── Dietary restrictions — zh ─────────────────────────────────
        {"input": p_rz + "我对花生过敏，哪些菜不能吃？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "花生过敏要注意，宫保鸡丁和部分凉菜会用到花生。下单时我会帮你标注过敏信息，避开这些菜。"},
        {"input": p_rz + "有没有无麸质的选择？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "大部分炒菜和蒸菜都是无麸质的，比如清蒸鱼、炒时蔬、白切鸡。面条类的需要避开。"},
        {"input": p_rz + "我是素食者，有什么推荐？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "推荐麻婆豆腐、炒时蔬、凉拌黄瓜和番茄蛋汤。如果全素的话，麻婆豆腐可以免肉末。"},
        {"input": p_re + "I'm allergic to shellfish. Is the fried rice safe?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Our vegetable fried rice is shellfish-free. I'd avoid the seafood fried rice though. I'll add a shellfish allergy note to your order."},

        # ── Reservations — zh ─────────────────────────────────────────
        {"input": p_rz + "我想订今天晚上七点的位子，三个人\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "好的，今晚七点，三位。是靠窗还是包间？我帮你查一下空位。"},
        {"input": p_rz + "周末有位子吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "周末比较满，建议提前两三天预订。你想订周六还是周日？几个人呢？"},
        {"input": p_re + "Can I book a table for four at 8pm tonight?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Let me check availability for tonight at 8pm for four. May I have your name for the reservation?"},

        # ── Complaints & feedback — zh ────────────────────────────────
        {"input": p_rz + "这道菜太咸了\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "非常抱歉！我马上通知后厨，给您重新做一份或者换一道菜，您看可以吗？"},
        {"input": p_rz + "等了很久了，菜还没上\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "实在不好意思，我帮您催一下厨房。今天人多，让您久等了。"},
        {"input": p_rz + "菜冷掉了\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "抱歉给您不好的体验。我马上安排厨房重新加热，或者给您重做一份。"},
        {"input": p_re + "This dish is way too salty.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "I'm sorry about that! Let me have the kitchen remake it for you right away, on the house."},

        # ── Payment & billing — zh ────────────────────────────────────
        {"input": p_rz + "买单\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "好的，您的账单总额是128元。需要我帮你叫服务员来结账吗？"},
        {"input": p_rz + "可以扫码支付吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "可以的，我们支持微信和支付宝扫码。服务员会把支付码拿过来。"},
        {"input": p_rz + "账单有点问题，多了一道菜\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "让我帮你核对一下订单记录。确实多了一道菜的话，我马上帮你修改账单。"},
        {"input": p_rz + "能开发票吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "可以的，结账时告诉服务员你的开票信息就行，电子发票会发到你手机上。"},

        # ── Recommendations — zh ─────────────────────────────────────
        {"input": p_rz + "两个人吃饭，推荐什么菜？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "两个人建议两菜一汤：宫保鸡丁配一个炒时蔬，再加番茄蛋汤，刚好不浪费。"},
        {"input": p_rz + "有没有适合小孩吃的？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "小朋友一般喜欢番茄炒蛋、糖醋里脊和玉米排骨汤，口味温和，吃着也放心。"},
        {"input": p_rz + "今天有什么特别的推荐吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "今天主厨推荐的清蒸鲈鱼很新鲜，是早上刚到的，配上姜葱蒸出来特别鲜美。"},
        {"input": p_re + "What would you recommend for a group of six?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "For six people, I'd suggest four main dishes, two sides, and a large soup. Our set menu C is popular for groups — it covers all the favorites."},

        # ── Mixed / edge cases — zh ───────────────────────────────────
        {"input": p_rz + "有没有Wi-Fi？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "有的，Wi-Fi密码在每张桌子的二维码旁边，扫一下就能看到。"},
        {"input": p_rz + "厕所在哪？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "洗手间在走廊尽头右转，男女分开的，标识很清楚。"},
        {"input": p_rz + "我能带宠物进来吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "抱歉，餐厅内不能带宠物。不过我们有室外的座位，天气好的话可以安排室外用餐。"},
        {"input": p_rz + "你们开到几点？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "我们中午十一点到下午两点，下午五点到晚上十点。晚上九点后不再接新单哦。"},
        {"input": p_rz + "谢谢，特别好吃\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "太好了，很高兴你喜欢！欢迎下次再来，记得提前预订。"},

        # ── Mixed — en ────────────────────────────────────────────────
        {"input": p_re + "Do you have gluten-free options?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Yes, most of our stir-fried and steamed dishes are gluten-free. I'd recommend the steamed fish and seasonal greens. Noodle dishes contain gluten, so I'd avoid those."},
        {"input": p_re + "Is there parking nearby?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "There's a parking lot behind the restaurant — first two hours are free with validation. Just ask your server to stamp your ticket."},
        {"input": p_re + "The food was excellent, thank you!\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "So glad you enjoyed it! We'd love to see you again. Don't forget to book ahead for weekends."},

        # =================================================================
        # C. CHEF / KDS — Kitchen Display System (40 examples, sys_chef_zh + sys_chef_en)
        # =================================================================

        # ── Order queue — zh ────────────────────────────────────────────
        {"input": p_kz + "现在有哪些待做的单子？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "当前待出菜：1号桌宫保鸡丁×1，3号桌麻婆豆腐×1、米饭×2，5号桌清蒸鲈鱼×1。宫保鸡丁优先，快好了。"},
        {"input": p_kz + "下一道要准备什么？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "按顺序下一道是6号桌的回锅肉，配一份米饭。4号桌的番茄蛋汤可以同时起锅，不耽误。"},
        {"input": p_kz + "帮我按桌号列出所有待做菜品\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "1号桌：宫保鸡丁（进行中）、米饭\n3号桌：麻婆豆腐、蛋花汤\n5号桌：清蒸鲈鱼\n6号桌：回锅肉、炒时蔬"},
        {"input": p_kz + "有多少张单子还没出？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "目前还有4张未出单，总共8道菜。高峰期注意节奏，热菜优先，凉菜可以稍后。"},
        {"input": p_kz + "哪桌等得最久？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "3号桌等了22分钟了，麻婆豆腐和蛋花汤还没出。建议优先处理3号桌。"},

        # ── Order queue — en ────────────────────────────────────────────
        {"input": p_ke + "Show me the pending orders.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Pending: Table 2 — kung pao chicken ×1, Table 4 — mapo tofu ×1, steamed rice ×2, Table 7 — steamed fish ×1. Kung pao is almost up."},
        {"input": p_ke + "What's up next?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Next is the twice-cooked pork for table 6. The tomato egg soup for table 4 can go out at the same time."},
        {"input": p_ke + "How many tickets are still open?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Five open tickets, 11 dishes total. Hot dishes first — cold appetizers can hold."},

        # ── Dish preparation & timing — zh ─────────────────────────────
        {"input": p_kz + "宫保鸡丁需要准备多久？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "宫保鸡丁从备料到出菜大概8分钟。鸡肉已经切好腌上了，下锅炒很快。"},
        {"input": p_kz + "清蒸鱼什么时候开始蒸？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "鱼已经杀好，等客人到店后再蒸，蒸8分钟就好。预计7点到，提前5分钟上蒸锅。"},
        {"input": p_kz + "这个牛腩煲要炖多久？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "牛腩煲慢炖需要90分钟，已经炖了60分钟，还剩半小时。5号桌刚好，时间对得上。"},
        {"input": p_kz + "几号桌的菜可以一起出？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "2号桌和6号桌都点了回锅肉，可以一锅出两份。另外3号桌和7号桌的米饭一起蒸。"},

        # ── Dish preparation — en ──────────────────────────────────────
        {"input": p_ke + "How long on the braised beef?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Braised beef needs 90 minutes total, 60 minutes in. Another 30 to go — right on time for table 5."},
        {"input": p_ke + "Can I fire table 3 and table 7 together?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Yes, both have stir-fry dishes. One wok for each, they'll be ready within a minute of each other."},

        # ── Customer arrival / reservation sync — zh ────────────────────
        {"input": p_kz + "今晚7点有多少预订？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "今晚7点有6桌预订，共22位客人。其中2桌预订时备注了有小孩，1桌要求靠窗座位。"},
        {"input": p_kz + "7点半那桌客人到了吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "还没到，已经迟了10分钟。如果7点45还没到，可以先把他们的凉菜撤单，热菜继续备料。"},
        {"input": p_kz + "下一批大单什么时候？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "8点有个8人聚会的大单，预订了套餐A。建议提前20分钟开始备料，凉菜可以先摆。"},
        {"input": p_kz + "今天中午的翻台情况怎么样？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "中午翻了两轮，共38桌。12点到1点是高峰，后厨压力最大。晚上预计会更多，做好准备。"},

        # ── Customer arrival — en ──────────────────────────────────────
        {"input": p_ke + "What's the cover count for tonight?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Tonight: 42 covers booked across 12 tables. Peak is 7pm with 6 tables arriving within 15 minutes of each other."},
        {"input": p_ke + "Has the 8-top for 7:30 arrived yet?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Not yet — they're 10 minutes late. If they don't show by 7:45, we can bump their hot dishes and hold the cold apps."},

        # ── Inventory / stock alerts — zh ──────────────────────────────
        {"input": p_kz + "哪些食材快用完了？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "鸡胸肉还剩2份的量，花生米只能再做3份宫保鸡丁。香菜和葱花也不多了，需要通知采购。"},
        {"input": p_kz + "牛肉还有多少？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "牛腩还有3份，牛柳剩2份。今晚预订里牛腩煲点了4份，不够了，建议联系供应商加急补货。"},
        {"input": p_kz + "检查一下调料库存\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "豆瓣酱剩半罐，生抽快见底了，花椒粉刚好够今晚。建议下单：生抽×2、豆瓣酱×1、花椒粉×1。"},
        {"input": p_kz + "鱼还够吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "鲈鱼今早进了8条，已经用了4条，还剩4条。今晚预订里还有3条要出，刚好够，不用补。"},
        {"input": p_kz + "有什么食材明天需要采购的？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "明天需要采购：鸡胸肉5kg、五花肉3kg、时令蔬菜各2份、鸡蛋2盘、豆腐4块。调料类生抽和料酒各补一瓶。"},

        # ── Inventory — en ──────────────────────────────────────────────
        {"input": p_ke + "What ingredients are running low?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Chicken breast: 2 portions left. Peanuts: enough for 3 more kung pao. Cilantro and scallions are low — notify purchasing."},
        {"input": p_ke + "How's the fish stock looking?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "8 sea bass came in this morning, 4 used. 3 more on tonight's books — exactly enough, no reorder needed."},
        {"input": p_ke + "Do we have enough tofu for tonight?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "4 blocks left, 3 orders of mapo tofu on the books. One spare — cutting it close but should be fine."},

        # ── Order modifications — zh ────────────────────────────────────
        {"input": p_kz + "3号桌的宫保鸡丁改成少辣\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "收到，3号桌宫保鸡丁少辣。已经在做的按原样出，新单子已更新为少辣。"},
        {"input": p_kz + "5号桌加一份米饭\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "5号桌加米饭已记录。跟之前的菜一起出，米饭已经蒸上了。"},
        {"input": p_kz + "8号桌取消了，单子撤掉\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "8号桌已取消。他们的清蒸鱼还没上锅，食材可以留给后面的单子用。其他备好的料先放冷藏。"},
        {"input": p_ke + "Table 4 wants the fish sauce on the side.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Noted — table 4 fish sauce on the side. I've updated the ticket so the pass won't miss it."},

        # ── Rush / priority — zh ───────────────────────────────────────
        {"input": p_kz + "7号桌客人催菜了\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "7号桌已标记为加急。他们点了水煮鱼和炒时蔬，水煮鱼已经在做了，炒时蔬马上起锅。"},
        {"input": p_kz + "VIP客人来了，优先出他们的菜\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "了解，VIP桌已置顶。他们的清蒸鲈鱼和牛腩煲优先处理，其他桌稍等几分钟。"},

        # ── Kitchen operations — zh ─────────────────────────────────────
        {"input": p_kz + "今天的备餐进度怎么样了？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "备餐进度：凉菜已全部备好，热菜备料完成了80%，蒸品区的鱼和排骨已经处理好。整体在计划内。"},
        {"input": p_kz + "还有多少桌没上齐菜？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "2号桌少一个汤，5号桌还差两个热菜，7号桌全齐了。重点关注2号和5号。"},
        {"input": p_kz + "后厨人手够不够？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "今天后厨3个人，炒锅2个、配菜1个。高峰期可能忙不过来，建议凉菜师傅来热菜区帮忙。"},
        {"input": p_kz + "今天总共有多少道菜？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "中午出了42道菜，晚上目前预订了58道。加上临时加的，今天预计120道左右，和昨天持平。"},
        {"input": p_ke + "What's the total dish count so far?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Lunch: 42 dishes out. Dinner: 58 on the books so far, plus walk-ins. On track for about 120 total today."},

        # ── Food safety / allergens — zh ────────────────────────────────
        {"input": p_kz + "2号桌客人花生过敏，注意\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "重要提醒：2号桌花生过敏。宫保鸡丁、凉拌菜和部分酱料含花生，出2号桌的菜务必检查。"},
        {"input": p_kz + "这道鱼确认是新鲜的吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "这条鲈鱼是今早6点到的，冷链运输，质检已通过。眼睛清亮、鳃红、肉质紧实，没问题。"},
    ]

    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--target", default="RK3576",
                        choices=["RK3576", "RK3588"],
                        help="Target Rockchip SoC (default: RK3576)")
    parser.add_argument("--quant", default="W4A16",
                        choices=["W4A16", "W8A8", "FP16"],
                        help="Quantisation mode (default: W4A16)")
    parser.add_argument("--max-context", type=int, default=4096,
                        help="Max context length in tokens (default: 4096)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Loading dtype (default: float16 — saves VRAM)")
    parser.add_argument("--calib-dataset", default=None,
                        help="Path to custom calibration JSON "
                        "(default: built-in 20-example set)")
    parser.add_argument("--out", default=None,
                        help="Output filename (default: auto-generated)")
    parser.add_argument("--gpu", default="0",
                        help="CUDA device index (default: 0)")
    args = parser.parse_args()

    # ── Import (may fail if not installed) ───────────────────────────
    try:
        from rkllm.api import RKLLM
    except ImportError:
        print("ERROR: RKLLM-Toolkit not installed.", file=sys.stderr)
        print("Download the SDK from https://console.zbox.filez.com/l/RJJDmB (code: rkllm)", file=sys.stderr)
        print("Then: pip install <path-to-wheel>", file=sys.stderr)
        return 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── Check CUDA ────────────────────────────────────────────────────
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"PyTorch CUDA: {'available' if cuda_ok else 'not available'}")
        if cuda_ok:
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except Exception:
        cuda_ok = False
        print("PyTorch CUDA: could not check")

    if not cuda_ok:
        print("WARNING: CUDA not detected. Conversion will run on CPU "
              "and may be very slow (hours instead of minutes).")
        print("On Vast.ai, verify you selected a GPU instance.")

    device = "cuda" if cuda_ok else "cpu"

    # ── Download model from HuggingFace ───────────────────────────────
    print(f"Downloading model: {args.model}")
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=args.model,
        local_dir=os.path.join(tempfile.gettempdir(),
                               args.model.replace("/", "_")),
        local_dir_use_symlinks=False,
    )
    print(f"  Model downloaded to: {local_dir}")

    # ── Calibration data ─────────────────────────────────────────────
    calib_path = args.calib_dataset
    if not calib_path:
        calib_path = os.path.join(tempfile.gettempdir(), "rkllm_calib.json")
        data = _calibration_data(args.model)
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(data)} calibration examples to {calib_path}")

    # ── NPU cores per platform ───────────────────────────────────────
    npu_cores = 3 if args.target == "RK3588" else 2

    # ── Quant algorithm ──────────────────────────────────────────────
    q_upper = args.quant.upper()
    if q_upper in ("W4A16",):
        quant_algo = "grq"
    else:
        quant_algo = "normal"

    # ── Output filename ──────────────────────────────────────────────
    model_slug = os.path.basename(args.model.rstrip("/")).lower().replace("/", "-")
    out = args.out or f"{model_slug}_{q_upper}_{args.target}.rkllm"

    # ── Load ─────────────────────────────────────────────────────────
    print(f"Loading model from: {local_dir}  (dtype={args.dtype}, device={device})")
    llm = RKLLM()
    ret = llm.load_huggingface(
        model=local_dir,
        model_lora=None,
        device=device,
        dtype=args.dtype,
        custom_config=None,
        load_weight=True,
    )
    if ret != 0:
        print(f"ERROR: load_huggingface returned {ret}", file=sys.stderr)
        return ret

    # ── Build ────────────────────────────────────────────────────────
    print(
        f"Building: quant={q_upper} algo={quant_algo} "
        f"target={args.target} npu_cores={npu_cores} ctx={args.max_context}"
    )
    ret = llm.build(
        do_quantization=(q_upper != "FP16"),
        optimization_level=1,
        quantized_dtype=q_upper if q_upper != "FP16" else "FP16",
        quantized_algorithm=quant_algo if q_upper != "FP16" else "normal",
        target_platform=args.target,
        num_npu_core=npu_cores,
        extra_qparams=None,
        dataset=calib_path,
        hybrid_rate=0,
        max_context=args.max_context,
    )
    if ret != 0:
        print(f"ERROR: build returned {ret}", file=sys.stderr)
        return ret

    # ── Export ───────────────────────────────────────────────────────
    print(f"Exporting → {out}")
    ret = llm.export_rkllm(out)
    if ret != 0:
        print(f"ERROR: export_rkllm returned {ret}", file=sys.stderr)
        return ret

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"Done — {out}  ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
