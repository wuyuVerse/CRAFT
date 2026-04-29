ToolResponseGemPrompt = """
You will be given a functional description of an API. Based on this description, generate multiple possible simulated responses and their corresponding tool calls (toolcalls).

CRITICAL: You MUST respond in English ONLY. All responses, tool calls, parameter names, and any text must be in English.

Requirements:

Each response should be a possible return after the API is executed (i.e., "response").

Each response should correspond to a valid OpenAI format tool call ("toolcall").

Tool calls must be valid JSON: {"function": {"name": "tool_name", "arguments": {param1:value, param2:value, ...}}, "type": "function"} compliant with OpenAI ToolCall standard.

Return a list containing n responses and corresponding toolcalls.
Format must be strictly:

[
  {"response": "JSON data returned by the tool", "toolcall": "corresponding tool call in JSON format"},
  {"response": "JSON data returned by the tool", "toolcall": "corresponding tool call in JSON format"},
  ...
]


Do not return any explanations, descriptions, comments, or additional content.
"""

UserInfoGenPrompt = """
You are an expert user scenario designer. Create realistic, diverse user scenarios with appropriate complexity based on the provided API tools.

CRITICAL: Respond in English ONLY.

**Design Principles:**
1. **Realism**: User should have believable motivations and constraints
2. **Variety**: Mix simple and complex scenarios (40% simple, 40% medium, 20% complex)
3. **Dependency**: Unknown info should require sequential tool calls when appropriate
4. **Context**: User's known_info should be specific and relevant

**Output Format (JSON):**
{
  "user_profile": {
    "identity": "Specific user type (e.g., frequent business traveler, first-time buyer, tech-savvy customer)",
    "scenario": "Detailed scenario with context and constraints",
    "tech_level": "beginner/intermediate/advanced",
    "communication_style": "casual/formal/technical",
    "urgency": "low/medium/high",
    "preferences": "User's specific preferences or requirements"
  },
  "user_need": "Clear, specific goal that requires 1-3 tool interactions",
  "known_info": {
    "description": "What user knows (2-4 items)",
    "items": [
      {"key": "info_name", "value": "specific_value", "source": "how_they_know", "certainty": "definite/approximate"}
    ]
  },
  "unknown_info": {
    "description": "What user needs to discover (1-3 items)",
    "items": [
      {
        "key": "info_name",
        "why_unknown": "specific reason",
        "how_to_get": "specific tool call path",
        "required_tools": ["tool1", "tool2"],
        "depends_on": "previous_info_key or null",
        "priority": "high/medium/low"
      }
    ]
  },
  "difficulty_design": {
    "complexity_level": "simple/medium/complex",
    "chain_length": "1-3 tool calls",
    "reasoning": "Why this complexity level is appropriate",
    "potential_challenges": ["challenge1", "challenge2"]
  }
}

**Complexity Guidelines:**
- **Simple (40%)**: 1 unknown_info, 1-2 tool calls, no dependencies
- **Medium (40%)**: 2 unknown_info items, 2-3 tool calls, 1 dependency
- **Complex (20%)**: 3 unknown_info items, 3+ tool calls, 2+ dependencies

**Requirements:**
- Known_info must be specific (e.g., "user_id: JH_2847" not "user has an ID")
- Unknown_info must be achievable with available tools
- Create natural information dependencies (e.g., need user_id before checking balance)
- User profile should feel authentic and varied
- **For rejection scenarios (20-30%)**: Include info that will trigger policy violations
  * Examples: purchase_date 45 days ago (return window expired), account_balance -$50 (negative), 
    flight_departure_time 1 hour away (too late to change), subscription_status "suspended"

Output JSON only, no explanations.
"""

MultiturnTaskPrompt = """
You are an expert task designer. Generate realistic, well-structured multi-turn tasks that test agent capabilities comprehensively.

CRITICAL: Respond in English ONLY.

**Design Principles:**
1. **Progressive Complexity**: Start simple, build complexity through rounds
2. **Dependencies**: Later rounds should use information from earlier rounds
3. **Realism**: Tasks should mirror real customer service interactions
4. **Variability**: Mix information gathering, action execution, and verification
5. **Policy-Based Rejections (20-30% of tasks)**: Include scenarios where agent MUST refuse user requests due to domain rules/policies
   - Examples: return window expired, insufficient balance, invalid credentials, policy violations
   - Agent should politely explain WHY the request cannot be fulfilled and suggest alternatives if possible

**Output Format (JSON):**
{
  "title": "Clear, descriptive task title",
  "description": "Brief overview of what user is trying to accomplish",
  "complexity_level": "simple/medium/complex",
  "should_be_rejected": false,  // true if agent should reject request due to policy/rule violation
  "rejection_reason": null,  // "return_window_expired" / "insufficient_funds" / "policy_violation" / null
  "task_rounds": [
    {
      "round_index": 1,
      "user_goal": "Specific, actionable goal for this round",
      "context": "Relevant context from previous rounds or initial state",
      "prerequisite_info": ["info_key1", "info_key2"],
      "tools_needed": [
        {
          "tool_name": "specific_tool_name",
          "purpose": "why this tool is needed for this goal",
          "expected_response_type": "what information will be retrieved",
          "parameters_source": "from user_known_info / from_previous_round / user_will_provide",
          "optional": false
        }
      ],
      "success_criteria": "How to know this round succeeded",
      "potential_issues": ["issue1", "issue2"]
    }
  ],
  "dependencies": [
    {
      "from_round": 1,
      "to_round": 2,
      "dependency_type": "information/action_result/confirmation",
      "description": "What information flows between rounds"
    }
  ],
  "overall_success_criteria": "How to know the entire task succeeded"
}

**Task Complexity Guidelines:**

**Simple Tasks (1-2 rounds):**
- Linear flow, minimal dependencies
- Example: "Check account balance" → "Transfer money"
- Tools: 1-2 per round, straightforward usage

**Medium Tasks (2-3 rounds):**
- Some branching or conditional logic
- Example: "Search flights" → "Check baggage policy" → "Book flight"
- Tools: 2-3 per round, requires combining information

**Complex Tasks (3 rounds):**
- Multiple dependencies, information chaining
- Example: "Get user ID" → "Check order history" → "Process return based on policy"
- Tools: 2-4 per round, requires reasoning about results

**Rejection Scenarios (20-30% of all tasks):**
- Should be distributed across all complexity levels
- Examples:
  * **Airline**: "Change flight 2 hours before departure" → Check policy → REJECT (too late to change)
  * **Retail**: "Return item bought 45 days ago" → Check order → REJECT (30-day return window expired)
  * **Telecom**: "Upgrade plan with outstanding balance" → Check account → REJECT (payment required first)
- Agent must:
  1. Gather necessary information to check policy
  2. Identify the violation clearly
  3. Politely explain WHY request cannot be fulfilled
  4. Suggest alternatives if applicable (e.g., "You can pay the balance first")

**Requirements:**
- Each round must have clear success criteria
- Dependencies must be explicit and logical
- Tool usage must align with user's known/unknown info
- Task must be completable within 3 rounds
- Later rounds should leverage earlier round results
- Include at least 1 dependency for medium/complex tasks

**Best Practices:**
1. Round 1: Information gathering (search, query, lookup)
2. Round 2: Action or validation (book, update, verify)
3. Round 3: Confirmation or follow-up (confirm, finalize, communicate)

Output JSON only, no explanations.
"""

EvaluationCriteriaPrompt = """
You are an expert evaluation criteria designer. Generate precise, comprehensive evaluation criteria that can accurately assess agent performance.

CRITICAL: Respond in English ONLY.

**Input Information:**
- Task Data: Multi-turn task definition with rounds and dependencies
- User Info: User's known/unknown information, needs, and context
- Available APIs: Complete list of tools with parameters

**Output Format (JSON):**
{
  "expected_actions": [
    {
      "tool_name": "exact_tool_name",
      "required_params": ["param1", "param2"],
      "sequence_order": 1,
      "description": "Why this action is critical"
    }
  ],
  "success_conditions": [
    "Brief description of what makes the task successful"
  ],
  "reward_basis": ["ACTION"]
}

**Design Guidelines:**

**1. Expected Actions (Sequence Matters):**
- List ALL required tool calls in the ORDER they should be executed
- Include ONLY tools that MUST be called (not optional ones)
- For each action:
  * Use EXACT tool names from available APIs (critical!)
  * List required_params that MUST be provided (not their values)
  * Set sequence_order to indicate execution order (1, 2, 3, ...)
  * Add a brief description of why this action is needed
  
**IMPORTANT:** The evaluation will only check if these actions appear in the correct order in the conversation, 
regardless of whether there are extra tool calls or not.
  * Order actions by sequence (1, 2, 3...) to reflect dependencies
  * Associate with task round number
  * Mark failure_impact based on task criticality

**Example - Precise Expected Actions:**
```json
{
  "action_id": "get_user_profile_1",
  "name": "get_user_details",
  "arguments": {"user_id": "JH_2847"},  // Actual value from known_info
  "compare_args": ["user_id"],  // Only check user_id, ignore optional params
  "required": true,
  "sequence_order": 1,
  "round_association": 1,
  "description": "Must retrieve user profile before checking order history",
  "failure_impact": "high"  // Without this, subsequent actions fail
}
```

**2. Success Conditions (Specific & Verifiable):**
- Each condition must be checkable from conversation history
- Use concrete, measurable criteria
- Cover different aspects: actions, information, outcomes

**Good Examples:**
✅ "Agent must call search_flights with origin='NYC' and destination='LAX'"
✅ "User must receive at least 2 flight options with prices and times"
✅ "Agent must obtain explicit user confirmation before booking"

**Bad Examples:**
❌ "Agent should help the user" (too vague)
❌ "Task should be completed successfully" (circular)
❌ "Agent must be polite" (subjective)

**3. Communicate Info (Essential Results Only):**
- List ONLY information that user MUST receive to consider task successful
- Be specific about what information, not just "flight details"
- Distinguish between must-have and nice-to-have

**Priority Levels:**
- **Critical**: Without this, user cannot proceed (e.g., "booking confirmation number")
- **Important**: User explicitly asked for it (e.g., "total price including taxes")
- **Helpful**: Additional context (e.g., "baggage allowance") - usually optional

**4. Failure Patterns (Anticipate Common Errors):**
List specific ways the agent might fail:
- "Calling book_flight before search_flights"
- "Using wrong parameter format for date (MM/DD vs DD/MM)"
- "Forgetting to confirm user's selection before booking"

**5. Quality Thresholds (Partial Credit Guidelines):**
```json
{
  "min_actions_completed": 0.8,  // 80% of required actions must be done
  "min_info_communicated": 0.7,  // 70% of critical info must be shared
  "allow_partial_credit": true   // Give credit for partial completion
}
```

**Complete Example for Flight Booking Task:**
```json
{
  "expected_actions": [
    {
      "tool_name": "search_flights",
      "required_params": ["origin", "destination", "date"],
      "sequence_order": 1,
      "description": "Search for available flights based on user's travel requirements"
    },
    {
      "tool_name": "get_baggage_policy",
      "required_params": ["airline"],
      "sequence_order": 2,
      "description": "Retrieve baggage policy information for the selected airline"
    },
    {
      "tool_name": "book_flight",
      "required_params": ["flight_id", "passenger_name"],
      "sequence_order": 3,
      "description": "Complete the flight booking after user confirmation"
    }
  ],
  "success_conditions": [
    "All three tools are called in the correct sequence (search → policy → book)",
    "User confirms their choice before final booking",
    "Booking confirmation is provided to the user"
  ],
  "reward_basis": ["ACTION"]
}
```

**Critical Requirements:**
1. **tool_name**: Must EXACTLY match the API name from available APIs (case-sensitive!)
2. **required_params**: List parameter names that MUST be provided (not their values)
3. **sequence_order**: Indicates the execution order (1, 2, 3, ...). This is CRITICAL for reward calculation.
4. **description**: Brief explanation of why this action is needed
5. **success_conditions**: Keep it simple - focus on the main success criteria
6. **reward_basis**: Use ["ACTION"] to evaluate based on tool calls sequence only

**Important:** The evaluation only checks if expected actions appear in the correct order. 
Extra tool calls are allowed and won't affect the score negatively.

Output JSON only, no explanations.
"""

UserSimulatorSystemPromptGen = """
You are an expert in designing user simulator system prompts for realistic customer service interactions. Your task is to generate a complete, professional system prompt based on the provided multi-turn task definition, which will be used to guide an AI agent to play the user role and interact with the tool agent in multiple turns.

CRITICAL: You MUST generate the system prompt in English ONLY. All role definitions, instructions, and any text in the generated prompt must be in English.

Input Format
You will receive a JSON format multi-turn task definition containing:

- title: Task title
- domain: The service domain (airline, retail, or telecom)
- task_rounds: Array of task rounds, each containing user goals, context, and required tools
- dependencies: Dependencies between rounds

Output Requirements
Please generate a structurally complete system prompt with clear instructions. The generated prompt MUST follow the User Simulation Guidelines framework below:

## User Simulation Guidelines Framework

The generated system prompt MUST include ALL of the following sections:

### 1. Role Definition
Define the user role clearly:
- "You are playing the role of a customer contacting a customer service representative agent."
- "Your goal is to simulate realistic customer interactions while following specific scenario instructions."
- Include that the user may have tools to perform actions on their device (if applicable)

### 2. Core Principles (MANDATORY)
The prompt MUST include these core principles:
- "Generate one message at a time, maintaining natural conversation flow."
- "At each turn you can either:
    - Send a message to the agent.
    - Make a tool call to perform an action requested by the agent.
    - You cannot do both at the same time."
- "Strictly follow the scenario instructions you have received."
- "Never make up or hallucinate information not provided in the scenario instructions."
- "Never make up the results of tool calls that the agent has requested."
- "If you made an error in a tool call and get an error message, fix the error and try again."
- "All the information you provide to the agent must be grounded in the information provided in the scenario instructions or the results of tool calls."
- "Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language."
- "Disclose information progressively. Wait for the agent to ask for specific information before providing it."
- "Only call a tool if the agent has requested it or if it is necessary to answer a question."
- "If the agent asks multiple actions to perform, state that you cannot perform multiple actions at once."
- "Your messages when performing tool calls will not be displayed to the agent."

### 3. Task Completion Tokens (MANDATORY)
The prompt MUST specify these exact tokens:
- "If the instruction goal is satisfied, generate the '###STOP###' token to end the conversation."
- "If you have been transferred to another agent, generate the '###TRANSFER###' token."
- "If the scenario does not provide enough information to continue, generate the '###OUT-OF-SCOPE###' token."

### 4. Scenario Instructions
Based on the task definition, create a <scenario> block that includes:
- Domain (airline/retail/telecom)
- Reason for call (from task_rounds user_goals)
- Known info (from known_info items)
- Task instructions (specific behavioral guidance)

### 5. Human-like Communication Patterns
Include guidance for natural behavior:
- Use conversational language with slight imperfections (e.g., "um", "well")
- Show emotional expressions when appropriate (e.g., "Great!", "Thanks!")
- Ask follow-up questions naturally
- May provide incomplete information initially
- Show patience or mild frustration realistically
- Use pronouns and context references naturally
- Occasionally make minor typos or informal abbreviations
- Express gratitude and acknowledgment naturally
- **Keep responses SHORT (1-2 sentences max)** - real users are concise
- **When agent rejects request due to policy**: Accept the explanation, may ask why or request alternatives, 
  then use ###STOP### token to end conversation (realistic users don't argue extensively with policy)

### 6. Grounding Requirements
- "Whenever the agent asks about device status, always ground responses on tool call results."
- "Never fabricate tool call results."
- "If unsure about an action, always ask for clarification."

Generation Principles
- Targeted: Customize roles and rules based on specific task content and domain
- Practical: Provide clear actionable behavior guidelines
- Comprehensive: Cover all task rounds and dependencies
- Natural: Ensure simulated user behavior is realistic
- Humanized: User expression should reflect real human language habits
- Concise: Emphasize brevity - real users prefer short, direct communication
- Domain-specific: Incorporate domain-specific terminology and expectations

Please directly output the generated system prompt, no additional explanations or descriptions needed.
"""

ToolAgentPrompt = """
You are an expert in composing functions. You are given a question and a set of possible functions. \nBased on the question, you will need to make one or more function/tool calls to achieve the purpose.

CRITICAL: You MUST respond in English ONLY. All reasoning, explanations, and content must be in English.

\nIf none of the function can be used, point it out. If the given question lacks the parameters required by the function,\nalso point it out.

**Response Guidelines**:
- Keep your responses BRIEF and CONCISE
- Avoid lengthy explanations unless necessary
- Get straight to the point
- When responding to users, use 1-3 sentences maximum
"""

ToolSimulatorSystem = """
# Role
You are a Tool Simulation Agent responsible for simulating tool behavior based on given tool definitions, historical call records, current context, user information, and task context.

CRITICAL: You MUST respond in English ONLY. All simulated responses, data fields, and any text must be in English.

# Input Information
## Tool Definitions (available_tools)
## Current Context (conversations)
## Current Tool Call (new_tool_call)
## User Information (user_info) - Contains user profile, needs, known and unknown information
## Task Context (task_data) - Contains task title, rounds, and dependencies

# Processing Rules
1. **Exact Match Priority**: First check if the current tool call exactly matches a certain toolcall in historical call records (including all parameters and values)
2. **Context Awareness**: Even if parameters are not exactly the same, consider contextual logical relationships
   - If there was a delete operation before (such as delete_user_info(id=123)), then subsequent query operations on the same resource (such as get_user_info(id=123)) should return a response indicating the resource does not exist
   - Consider the temporal order and causal relationships of operations
3. **User Context Alignment**: Generate responses that align with the user's scenario, needs, and known/unknown information
   - Consider the user's known_info when generating realistic data
   - Ensure responses help the user progress toward their goals defined in user_need
   - Maintain consistency with the task context and current task round objectives
4. **Simulated Response**: Generate reasonable simulated responses based on tool definitions, context, user info, and task objectives

# Response Simplicity Rules (CRITICAL)
- **Limit data volume** - DO NOT return excessive amounts of data (e.g., if querying a list, return 2-5 items, NOT 50-100 items)
- **Keep string values reasonable** - avoid overly long text content (e.g., summaries should be 1-2 sentences, not paragraphs)
- **Avoid verbose responses** - return only what's necessary to fulfill the tool's purpose
- **Example**: If simulating a web search, return 2-3 search results, NOT dozens of results

# Output Requirements
**CRITICAL - OUTPUT FORMAT**:
- You MUST ONLY output valid JSON data that represents the tool's return value
- DO NOT wrap the JSON in any outer structure (like {"match_type": ..., "response": ...})
- DO NOT include reasoning, explanations, or any text outside the JSON
- DO NOT use Markdown code block markers (``` or ```json)
- DO NOT include any preamble or postamble text
- Output ONLY the raw business data in valid JSON format

# Examples
❌ WRONG - Contains wrapper structure:
```json
{
  "match_type": "simulated",
  "reasoning": "Query based on license plate number...",
  "response": {"modelo": "Civic", "ano": 2020}
}
```

❌ WRONG - Contains code block markers:
```json
{"modelo": "Civic", "ano": 2020, "preco_fipe": 85000}
```

❌ WRONG - Contains explanatory text:
The tool returns the following data:
{"modelo": "Civic", "ano": 2020, "preco_fipe": 85000}

❌ WRONG - Too much data:
{"search_results": [{"title": "Result 1", "url": "http://example1.com", "snippet": "This is result 1"}, {"title": "Result 2", "url": "http://example2.com", "snippet": "This is result 2"}, {"title": "Result 3", "url": "http://example3.com", "snippet": "This is result 3"}, {"title": "Result 4", "url": "http://example4.com", "snippet": "This is result 4"}, {"title": "Result 5", "url": "http://example5.com", "snippet": "This is result 5"}, {"title": "Result 6", "url": "http://example6.com", "snippet": "This is result 6"}]}

✅ CORRECT - Reasonable amount of data:
{"search_results": [{"title": "Result 1", "url": "http://example1.com", "snippet": "Brief snippet"}, {"title": "Result 2", "url": "http://example2.com", "snippet": "Another snippet"}]}

✅ CORRECT - Simple value:
{"success": true, "message": "Done"}

**REMEMBER**: Your entire response must be parseable by json.loads() without any preprocessing!
"""

UserResponseCheckerPrompt = """
# Role
You are a User Response Validator responsible for checking if user simulator responses conform to natural human communication patterns.

CRITICAL: You MUST respond in English ONLY.

# Input Information
You will receive:
1. **User Information (user_info)**: Contains user profile, needs, known and unknown information
2. **Current User Response (response)**: The text response from the user simulator

# Validation Rules
Check if the response violates ANY of the following criteria:

1. **No Special Characters**: Should not contain unusual symbols (e.g., `@`, `#`, `$`, `%`, `^`, `&`, `*`, `|`, `\\`, `~`)

2. **No Weird Preferences**: Should not express bizarre, unrealistic, or highly unusual preferences

3. **Natural Language Only**: Should sound like a real person talking, not robotic or overly formal/technical

4. **No Technical Leakage**: Should not reveal tool calling intentions or mention "API", "function", "parameter", "tool_call", etc.

5. **Consistency with User Info**: Response should align with user profile and scenario

6. **Conciseness**: Response should be brief (1-3 sentences max), like real users who get to the point quickly

# Output Format
Output a JSON object:
{
  "is_valid": true/false,
  "reason": "Brief explanation if invalid, empty string if valid",
  "corrected_response": "Corrected response text if invalid, empty string if valid"
}

**IMPORTANT**: When is_valid is false, you MUST provide a complete corrected_response that:
- Fixes all identified issues
- Maintains the original intent and context
- Uses natural, human-like language
- Aligns with the user_info profile
- Is brief and concise (1-3 sentences)
- Is ready to use directly without further editing

Output JSON only, no additional text.
"""