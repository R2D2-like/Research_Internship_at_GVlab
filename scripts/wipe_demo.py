import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

def main():
    # コントローラの設定を読み込み
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # 環境を作成（wipeタスク）
    env = suite.make(
        env_name="Wipe",  # 環境名
        robots="UR5e",   # ロボットの種類
        controller_configs=controller_config,
        has_renderer=True,   # レンダラーを有効にする（GUIで可視化）
        render_camera="frontview",  # カメラの視点
        has_offscreen_renderer=False,  # Off-screen renderer を無効にする
        use_camera_obs=False,
    )

    # 環境をGymのラッパーで包む（オプション）
    env = GymWrapper(env)

    # 環境のリセット
    obs = env.reset()

    # デモのためにランダムな動作を100ステップ実行
    for _ in range(100):
        action = env.action_space.sample()  # ランダムなアクションを選択
        obs, reward, done, _, info = env.step(action)  # アクションを環境に適用
        if env.has_renderer:
            env.render()  # GUIで状態を表示

        if done:
            break

if __name__ == "__main__":
    main()
