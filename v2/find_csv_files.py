# find_csv_files.py
import os
import glob

def find_all_csv_files():
    """모든 CSV 파일을 찾아서 출력"""
    print("🔍 CSV 파일 찾기 도구")
    print("=" * 40)
    
    # 현재 위치 확인
    current_dir = os.getcwd()
    print(f"📍 현재 디렉토리: {current_dir}")
    
    # 1. 현재 디렉토리
    print(f"\n📁 현재 디렉토리 내용:")
    current_files = os.listdir('.')
    csv_files = [f for f in current_files if f.endswith('.csv')]
    dirs = [d for d in current_files if os.path.isdir(d)]
    
    if csv_files:
        print(f"   CSV 파일들:")
        for csv_file in csv_files:
            size = os.path.getsize(csv_file) / 1024
            print(f"      ✅ {csv_file} ({size:.1f} KB)")
    else:
        print(f"   ❌ CSV 파일 없음")
    
    if dirs:
        print(f"   디렉토리들: {dirs}")
    
    # 2. 상위 디렉토리
    print(f"\n📁 상위 디렉토리 (..) 내용:")
    try:
        parent_files = os.listdir('..')
        parent_csv = [f for f in parent_files if f.endswith('.csv')]
        parent_dirs = [d for d in parent_files if os.path.isdir(os.path.join('..', d))]
        
        if parent_csv:
            print(f"   CSV 파일들:")
            for csv_file in parent_csv:
                try:
                    size = os.path.getsize(os.path.join('..', csv_file)) / 1024
                    print(f"      ✅ ../{csv_file} ({size:.1f} KB)")
                except:
                    print(f"      ✅ ../{csv_file}")
        else:
            print(f"   ❌ CSV 파일 없음")
        
        print(f"   디렉토리들: {parent_dirs}")
        
        # csi_data 폴더 확인
        if 'csi_data' in parent_dirs:
            print(f"\n📂 ../csi_data 폴더 탐색:")
            csi_data_path = '../csi_data'
            
            try:
                csi_subdirs = [d for d in os.listdir(csi_data_path) if os.path.isdir(os.path.join(csi_data_path, d))]
                print(f"   하위 디렉토리들: {csi_subdirs}")
                
                for subdir in csi_subdirs[:5]:  # 최대 5개만
                    subdir_path = os.path.join(csi_data_path, subdir)
                    try:
                        subdir_files = os.listdir(subdir_path)
                        subdir_csv = [f for f in subdir_files if f.endswith('.csv')]
                        
                        if subdir_csv:
                            print(f"   📁 {subdir}/ ({len(subdir_csv)}개 CSV):")
                            for csv_file in subdir_csv[:3]:  # 최대 3개만 표시
                                csv_path = os.path.join(subdir_path, csv_file)
                                try:
                                    size = os.path.getsize(csv_path) / 1024
                                    print(f"      ✅ {csi_data_path}/{subdir}/{csv_file} ({size:.1f} KB)")
                                except:
                                    print(f"      ✅ {csi_data_path}/{subdir}/{csv_file}")
                            if len(subdir_csv) > 3:
                                print(f"      ... 외 {len(subdir_csv)-3}개")
                    except:
                        print(f"   ❌ {subdir}/ 접근 불가")
            except:
                print(f"   ❌ csi_data 폴더 접근 불가")
        
    except:
        print(f"   ❌ 상위 디렉토리 접근 불가")
    
    # 3. 전체 검색 (재귀적)
    print(f"\n🔍 전체 검색 (재귀적):")
    
    search_patterns = [
        "./*.csv",
        "../*.csv", 
        "../*/*.csv",
        "../*/*/*.csv",
        "./v2/*.csv",
        "../csi_data/*.csv",
        "../csi_data/*/*.csv",
        "../csi_data/*/*/*.csv"
    ]
    
    found_files = []
    for pattern in search_patterns:
        try:
            files = glob.glob(pattern)
            if files:
                print(f"   패턴 '{pattern}': {len(files)}개 발견")
                for file_path in files[:3]:  # 최대 3개만 표시
                    try:
                        size = os.path.getsize(file_path) / 1024
                        print(f"      ✅ {file_path} ({size:.1f} KB)")
                        found_files.append(file_path)
                    except:
                        print(f"      ✅ {file_path}")
                        found_files.append(file_path)
                if len(files) > 3:
                    print(f"      ... 외 {len(files)-3}개")
        except Exception as e:
            print(f"   패턴 '{pattern}': 검색 실패 ({e})")
    
    # 4. 요약
    print(f"\n📋 검색 요약:")
    if found_files:
        print(f"   총 발견된 CSV 파일: {len(set(found_files))}개")
        
        # 추천 파일 (크기 기준)
        try:
            file_sizes = []
            for file_path in set(found_files):
                try:
                    size = os.path.getsize(file_path)
                    file_sizes.append((file_path, size))
                except:
                    pass
            
            if file_sizes:
                file_sizes.sort(key=lambda x: x[1], reverse=True)
                print(f"\n🎯 추천 파일 (크기 순):")
                for i, (file_path, size) in enumerate(file_sizes[:5]):
                    size_kb = size / 1024
                    print(f"   {i+1}. {file_path} ({size_kb:.1f} KB)")
                
                return [fp for fp, _ in file_sizes[:5]]
        except:
            pass
    else:
        print(f"   ❌ CSV 파일을 찾을 수 없습니다!")
    
    return found_files

def test_with_found_file():
    """찾은 파일로 바로 테스트"""
    found_files = find_all_csv_files()
    
    if found_files:
        print(f"\n🚀 첫 번째 파일로 모델 테스트:")
        test_file = found_files[0]
        print(f"   선택된 파일: {test_file}")
        
        try:
            from model_tester import test_specific_file
            test_specific_file(test_file)
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            print(f"💡 수동으로 테스트하려면:")
            print(f"   python -c \"from model_tester import test_specific_file; test_specific_file('{test_file}')\"")
    else:
        print(f"\n❌ 테스트할 파일이 없습니다!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_with_found_file()
    else:
        find_all_csv_files()