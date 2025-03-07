from utils.neodb import app
import traceback
class PendingRelationships:
    def __init__(self):
        self.pending_relationships = []

    def add_relationship(self, parent_id, target_id, relation):

        relationship = {
            "parent_id": parent_id,
            "target_id": target_id,
            "relation": relation
        }
        self.pending_relationships.append(relationship)

    def get_pending_relationships(self):
        return self.pending_relationships

    def clear_relationships(self):
        self.pending_relationships = []

    def create_all_relationships(self):
        success_count = 0
        failure_count = 0
        
        print(" the pending relations are :::")
        for rel in self.pending_relationships:
            try:
                if(app.get_node_by_id(rel['target_id'])):
                    print(rel)
                    app.create_relation(
                        child_id=rel['parent_id'],
                        parent_id=rel['target_id'],
                        relation=rel['relation']
                    )
                    success_count += 1
            except Exception as e:
                failure_count += 1
                print(f"❌ Failed to create relationship {rel}: {str(e)}")
                traceback.print_exc()

        print(f"Created {success_count} relationships successfully")
        if failure_count > 0:
            print(f"Failed to create {failure_count} relationships")

pending_rels = PendingRelationships()